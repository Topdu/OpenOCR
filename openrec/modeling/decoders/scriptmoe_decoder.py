"""
Script-Aware Mixture-of-Experts (ScriptMoE) Decoder for Multilingual Text Recognition.

Key Innovation:
    Different writing systems (scripts) exhibit fundamentally different visual patterns
    and decoding logic. Instead of using a single shared decoder, we introduce a
    Script-Aware MoE mechanism that routes different scripts to specialized experts
    while maintaining a shared expert for cross-script knowledge transfer.

Architecture:
    Encoder (SVTRv2, shared)
        → Script Router (lightweight, predicts script distribution from encoder features)
        → MoE Decoder Layers:
            - N script-specialized experts (FFN experts)
            - 1 shared expert (always activated)
            - Top-K gating
        → Output projection

Reference:
    - Switch Transformer (Fedus et al., 2022) for MoE gating
    - SMoE (Zuo et al., 2022) for shared expert design
"""

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from openrec.modeling.common import Mlp


# ============================================================================
# Expert FFN Module
# ============================================================================
class ExpertFFN(nn.Module):
    """A single expert Feed-Forward Network (FFN).

    Each expert has its own independent parameters, allowing specialization
    for specific script families.
    """

    def __init__(self, d_model, dim_feedforward, dropout=0.1, act_layer=nn.ReLU):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.act = act_layer()
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [B*T, D] or [num_tokens, D]
        Returns:
            [B*T, D] or [num_tokens, D]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ============================================================================
# Script-Aware Router
# ============================================================================
class ScriptAwareRouter(nn.Module):
    """Computes routing weights for each token to each expert.

    Supports two routing modes:
    1. 'token': Each token independently selects top-K experts (fine-grained).
    2. 'sample': All tokens in a sample share the same routing (script-level).

    The router takes encoder memory features to predict script distribution,
    enabling script-aware routing without explicit script labels during inference.
    """

    def __init__(
        self,
        d_model,
        num_experts,
        top_k=2,
        routing_mode='sample',
        jitter_noise=0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.routing_mode = routing_mode
        self.jitter_noise = jitter_noise

        # Router network: project features to expert logits
        self.router = nn.Linear(d_model, num_experts, bias=False)
        nn.init.kaiming_uniform_(self.router.weight, a=math.sqrt(5))

    def forward(self, x, memory=None):
        """
        Args:
            x: decoder hidden states [B, T, D]
            memory: encoder output [B, N, D], used for script-level routing

        Returns:
            router_probs: [B, T, num_experts] or [B, 1, num_experts]
            top_k_indices: [B, T, top_k] or [B, 1, top_k]
            top_k_weights: [B, T, top_k] or [B, 1, top_k]
            router_logits: [B, T, num_experts]
        """
        if self.routing_mode == 'sample' and memory is not None:
            # Sample-level routing: use mean-pooled encoder features
            # This ensures all tokens in one sample go to the same experts
            routing_input = memory.mean(dim=1, keepdim=True)  # [B, 1, D]
        else:
            # Token-level routing: each decoder token routes independently
            routing_input = x  # [B, T, D]

        # Add jitter noise during training for better load balancing
        if self.training and self.jitter_noise > 0:
            routing_input = routing_input * (
                1.0 + torch.randn_like(routing_input) * self.jitter_noise
            )

        router_logits = self.router(routing_input)  # [B, ?, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-K selection
        top_k_weights, top_k_indices = torch.topk(
            router_probs, self.top_k, dim=-1
        )
        # Re-normalize top-k weights
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-9)

        return router_probs, top_k_indices, top_k_weights, router_logits


# ============================================================================
# MoE FFN Layer (replaces standard FFN in TransformerBlock)
# ============================================================================
class MoEFFNLayer(nn.Module):
    """Mixture-of-Experts FFN layer with a shared expert.

    Replaces the standard FFN in transformer decoder blocks. Contains:
    - N specialized experts (only top-K are activated per sample)
    - 1 shared expert (always activated for all samples)

    The shared expert ensures cross-script knowledge transfer, especially
    beneficial for low-resource scripts like Tibetan.
    """

    def __init__(
        self,
        d_model,
        dim_feedforward,
        num_experts=4,
        top_k=2,
        dropout=0.1,
        routing_mode='sample',
        jitter_noise=0.1,
        shared_expert_ratio=1.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.d_model = d_model

        # Script-aware router
        self.router = ScriptAwareRouter(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            routing_mode=routing_mode,
            jitter_noise=jitter_noise,
        )

        # Specialized experts
        self.experts = nn.ModuleList([
            ExpertFFN(d_model, dim_feedforward, dropout)
            for _ in range(num_experts)
        ])

        # Shared expert (always activated)
        shared_ff_dim = int(dim_feedforward * shared_expert_ratio)
        self.shared_expert = ExpertFFN(d_model, shared_ff_dim, dropout)

        # Learnable weight for combining shared and routed outputs
        self.shared_gate = nn.Linear(d_model, 1, bias=False)

    def forward(self, x, memory=None):
        """
        Args:
            x: [B, T, D] decoder hidden states
            memory: [B, N, D] encoder output (for routing)

        Returns:
            output: [B, T, D]
            aux_data: dict with routing info (router logits/probs, expert indices)
        """
        B, T, D = x.shape

        # Get routing decisions
        router_probs, top_k_indices, top_k_weights, router_logits = \
            self.router(x, memory)

        # Expand sample-level routing to token-level if needed
        if top_k_indices.shape[1] == 1 and T > 1:
            top_k_indices = top_k_indices.expand(-1, T, -1)  # [B, T, top_k]
            top_k_weights = top_k_weights.expand(-1, T, -1)  # [B, T, top_k]

        # ---- Routed expert computation ----
        # Flatten for easier indexing
        x_flat = x.reshape(B * T, D)  # [BT, D]
        indices_flat = top_k_indices.reshape(B * T, self.top_k)  # [BT, top_k]
        weights_flat = top_k_weights.reshape(B * T, self.top_k)  # [BT, top_k]

        if self.training:
            # Training keeps the original sparse (masked) dispatch to preserve
            # exact behaviour / gradients of trained checkpoints.
            routed_output = torch.zeros_like(x)  # [B, T, D]
            for k in range(self.top_k):
                expert_indices = indices_flat[:, k]  # [BT]
                expert_weights = weights_flat[:, k].unsqueeze(-1)  # [BT, 1]

                for e_idx in range(self.num_experts):
                    mask = (expert_indices == e_idx)  # [BT]
                    if mask.any():
                        expert_input = x_flat[mask]  # [num_tokens, D]
                        expert_output = self.experts[e_idx](expert_input)
                        routed_output.view(B * T, D)[mask] += (
                            expert_output * expert_weights[mask]
                        )
        else:
            # Inference fast path: sync-free dense dispatch.
            # Scatter the top-k gate weights into a dense [BT, E] weight
            # matrix, then run each expert on all tokens weighted by its
            # column.  This removes the per-expert ``mask.any()`` host
            # synchronisation and the dynamic-shape gather/scatter, which
            # dominate latency at low batch size.  Numerically identical to
            # the sparse version (unselected experts get weight 0).
            gate_full = torch.zeros(
                B * T, self.num_experts, device=x.device, dtype=x.dtype,
            )
            gate_full.scatter_(1, indices_flat, weights_flat)  # [BT, E]

            routed_flat = torch.zeros_like(x_flat)  # [BT, D]
            for e_idx in range(self.num_experts):
                w_e = gate_full[:, e_idx:e_idx + 1]  # [BT, 1]
                routed_flat = routed_flat + self.experts[e_idx](x_flat) * w_e
            routed_output = routed_flat.view(B, T, D)

        # ---- Shared expert computation ----
        shared_output = self.shared_expert(x)  # [B, T, D]

        # ---- Combine shared and routed outputs ----
        shared_weight = torch.sigmoid(self.shared_gate(x))  # [B, T, 1]
        output = shared_weight * shared_output + (1 - shared_weight) * routed_output

        # Collect routing info (router logits/probs, expert indices)
        aux_data = {
            'router_logits': router_logits,
            'router_probs': router_probs,
            'top_k_indices': top_k_indices,
        }

        return output, aux_data


# ============================================================================
# MoE Attention (reuse from nrtr_decoder)
# ============================================================================
class MultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0.0, self_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, \
            'embed_dim must be divisible by num_heads'
        self.scale = self.head_dim ** -0.5
        self.self_attn = self_attn
        if self_attn:
            self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        else:
            self.q = nn.Linear(embed_dim, embed_dim)
            self.kv = nn.Linear(embed_dim, embed_dim * 2)
        self.attn_drop = nn.Dropout(dropout)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key=None, attn_mask=None):
        B, qN = query.shape[:2]
        if self.self_attn:
            qkv = self.qkv(query)
            qkv = qkv.reshape(B, qN, 3, self.num_heads,
                              self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
        else:
            kN = key.shape[1]
            q = self.q(query)
            q = q.reshape(B, qN, self.num_heads, self.head_dim).transpose(1, 2)
            kv = self.kv(key)
            kv = kv.reshape(B, kN, 2, self.num_heads,
                            self.head_dim).permute(2, 0, 3, 1, 4)
            k, v = kv.unbind(0)

        attn = (q.matmul(k.transpose(2, 3))) * self.scale
        if attn_mask is not None:
            attn += attn_mask
        attn = F.softmax(attn, dim=-1)
        if not self.training:
            self.attn_map = attn
        attn = self.attn_drop(attn)
        x = (attn.matmul(v)).transpose(1, 2)
        x = x.reshape(B, qN, self.embed_dim)
        x = self.out_proj(x)
        return x


# ============================================================================
# MoE Transformer Decoder Block
# ============================================================================
class MoETransformerBlock(nn.Module):
    """Transformer Decoder Block with MoE FFN.

    Structure:
        Self-Attention → Cross-Attention → MoE FFN
    The FFN is replaced with a Mixture-of-Experts layer that routes
    tokens to script-specialized experts.
    """

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        attention_dropout_rate=0.0,
        residual_dropout_rate=0.1,
        num_experts=4,
        top_k=2,
        routing_mode='sample',
        jitter_noise=0.1,
        shared_expert_ratio=1.0,
        with_self_attn=True,
        with_cross_attn=True,
        epsilon=1e-5,
    ):
        super().__init__()

        # Self-Attention
        self.with_self_attn = with_self_attn
        if with_self_attn:
            self.self_attn = MultiheadAttention(
                d_model, nhead,
                dropout=attention_dropout_rate,
                self_attn=True,
            )
            self.norm1 = nn.LayerNorm(d_model, eps=epsilon)
            self.dropout1 = nn.Dropout(residual_dropout_rate)

        # Cross-Attention
        self.with_cross_attn = with_cross_attn
        if with_cross_attn:
            self.cross_attn = MultiheadAttention(
                d_model, nhead,
                dropout=attention_dropout_rate,
                self_attn=False,
            )
            self.norm2 = nn.LayerNorm(d_model, eps=epsilon)
            self.dropout2 = nn.Dropout(residual_dropout_rate)

        # MoE FFN (replaces standard FFN)
        self.moe_ffn = MoEFFNLayer(
            d_model=d_model,
            dim_feedforward=dim_feedforward,
            num_experts=num_experts,
            top_k=top_k,
            dropout=residual_dropout_rate,
            routing_mode=routing_mode,
            jitter_noise=jitter_noise,
            shared_expert_ratio=shared_expert_ratio,
        )
        self.norm3 = nn.LayerNorm(d_model, eps=epsilon)
        self.dropout3 = nn.Dropout(residual_dropout_rate)

    def forward(self, tgt, memory=None, self_mask=None, cross_mask=None):
        """
        Args:
            tgt: [B, T, D] target sequence
            memory: [B, N, D] encoder output
            self_mask: causal mask for self-attention
            cross_mask: mask for cross-attention

        Returns:
            tgt: [B, T, D]
            aux_data: dict with MoE routing info
        """
        # Self-Attention
        if self.with_self_attn:
            tgt1 = self.self_attn(tgt, attn_mask=self_mask)
            tgt = self.norm1(tgt + self.dropout1(tgt1))

        # Cross-Attention
        if self.with_cross_attn:
            tgt2 = self.cross_attn(tgt, key=memory, attn_mask=cross_mask)
            tgt = self.norm2(tgt + self.dropout2(tgt2))

        # MoE FFN
        ffn_out, aux_data = self.moe_ffn(tgt, memory=memory)
        tgt = self.norm3(tgt + self.dropout3(ffn_out))

        return tgt, aux_data


# ============================================================================
# Positional Encoding (reuse from nrtr_decoder)
# ============================================================================
class PositionalEncoding(nn.Module):

    def __init__(self, dropout, dim, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros([max_len, dim])
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = torch.unsqueeze(pe, 0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)


class Embeddings(nn.Module):

    def __init__(self, d_model, vocab, padding_idx=None, scale_embedding=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab, d_model, padding_idx=padding_idx)
        self.embedding.weight.data.normal_(mean=0.0, std=d_model ** -0.5)
        self.d_model = d_model
        self.scale_embedding = scale_embedding

    def forward(self, x):
        if self.scale_embedding:
            x = self.embedding(x)
            return x * math.sqrt(self.d_model)
        return self.embedding(x)


# ============================================================================
# ScriptMoE Decoder (main class)
# ============================================================================
class ScriptMoEDecoder(nn.Module):
    """Script-Aware Mixture-of-Experts Decoder for Multilingual Text Recognition.

    Extends the standard NRTRDecoder by replacing the FFN in each decoder layer
    with a Mixture-of-Experts layer. A lightweight script router determines
    which experts to activate based on encoder features.

    Key features:
    1. Script-specialized experts: Each expert learns patterns specific to a
       script family (e.g., Latin/Cyrillic, CJK, Arabic/RTL, Indic/Thai).
    2. Shared expert: Always activated, captures cross-script common knowledge.
    3. Sample-level routing: All tokens in one image share the same expert set,
       since a text image typically contains only one script.

    Args:
        in_channels (int): Input feature dimension from encoder.
        out_channels (int): Number of output classes (vocab size).
        nhead (int): Number of attention heads.
        num_encoder_layers (int): Number of encoder layers (-1 to skip).
        num_decoder_layers (int): Number of decoder layers.
        num_experts (int): Number of specialized experts.
        top_k (int): Number of experts activated per sample.
        routing_mode (str): 'sample' or 'token' level routing.
        jitter_noise (float): Noise for router during training.
        shared_expert_ratio (float): Ratio of shared expert FFN dimension.
        max_len (int): Maximum decoding length.
        beam_size (int): Beam size (0 for greedy decoding).
        attention_dropout_rate (float): Dropout for attention.
        residual_dropout_rate (float): Dropout for residual connections.
        scale_embedding (bool): Whether to scale embeddings.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        nhead=None,
        num_encoder_layers=-1,
        num_decoder_layers=2,
        num_experts=4,
        top_k=2,
        routing_mode='sample',
        jitter_noise=0.1,
        shared_expert_ratio=1.0,
        expert_ffn_ratio=4.0,
        max_len=25,
        beam_size=0,
        attention_dropout_rate=0.0,
        residual_dropout_rate=0.1,
        scale_embedding=True,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.ignore_index = out_channels - 1
        self.bos = out_channels - 2
        self.eos = 0
        self.max_len = max_len
        self.num_experts = num_experts
        self.top_k = top_k

        d_model = in_channels
        # FFN hidden size for each expert (specialized & shared base size).
        # Default 4x preserves the original capacity; use a larger ratio
        # (e.g. 16/32) to scale up expert capacity for "fat-expert" MoE.
        dim_feedforward = int(d_model * expert_ffn_ratio)
        nhead = nhead if nhead is not None else d_model // 32

        # Token embedding and positional encoding
        self.embedding = Embeddings(
            d_model=d_model,
            vocab=self.out_channels,
            padding_idx=0,
            scale_embedding=scale_embedding,
        )
        self.positional_encoding = PositionalEncoding(
            dropout=residual_dropout_rate, dim=d_model,
        )

        # Optional encoder layers (usually disabled, i.e., num_encoder_layers=-1)
        if num_encoder_layers > 0:
            from openrec.modeling.decoders.nrtr_decoder import TransformerBlock
            self.encoder = nn.ModuleList([
                TransformerBlock(
                    d_model, nhead, dim_feedforward,
                    attention_dropout_rate, residual_dropout_rate,
                    with_self_attn=True, with_cross_attn=False,
                ) for _ in range(num_encoder_layers)
            ])
        else:
            self.encoder = None

        # MoE Decoder layers
        self.decoder = nn.ModuleList([
            MoETransformerBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                attention_dropout_rate=attention_dropout_rate,
                residual_dropout_rate=residual_dropout_rate,
                num_experts=num_experts,
                top_k=top_k,
                routing_mode=routing_mode,
                jitter_noise=jitter_noise,
                shared_expert_ratio=shared_expert_ratio,
                with_self_attn=True,
                with_cross_attn=True,
            ) for _ in range(num_decoder_layers)
        ])

        # Script classifier head (auxiliary task for explicit supervision)
        # Uses encoder features to predict script type
        self.script_classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_experts),
        )

        # Output projection
        self.beam_size = beam_size
        self.d_model = d_model
        self.nhead = nhead
        self.tgt_word_prj = nn.Linear(d_model, self.out_channels - 2, bias=False)

        w0 = np.random.normal(
            0.0, d_model ** -0.5,
            (d_model, self.out_channels - 2)
        ).astype(np.float32)
        self.tgt_word_prj.weight.data = torch.from_numpy(w0.transpose())

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _encode_memory(self, src):
        """Optionally pass through encoder layers."""
        if self.encoder is not None:
            src = self.positional_encoding(src)
            for encoder_layer in self.encoder:
                src = encoder_layer(src)
        return src

    def forward_train(self, src, tgt):
        """Training forward pass.

        Args:
            src: [B, N, D] encoder output
            tgt: [B, max_len+2] target token indices (BOS + text + EOS + PAD)

        Returns:
            logit: [B, T, vocab_size-2] prediction logits
            aux_data_list: list of aux_data dicts from each decoder layer
            script_logits: [B, num_experts] script classification logits
        """
        tgt = tgt[:, :-1]  # Remove last token (teacher forcing)

        tgt = self.embedding(tgt)
        tgt = self.positional_encoding(tgt)
        tgt_mask = self.generate_square_subsequent_mask(
            tgt.shape[1], device=src.device)

        memory = self._encode_memory(src)

        # Script classification from encoder features
        script_logits = self.script_classifier(memory.mean(dim=1))  # [B, num_experts]

        # Decode with MoE layers
        aux_data_list = []
        for decoder_layer in self.decoder:
            tgt, aux_data = decoder_layer(tgt, memory, self_mask=tgt_mask)
            aux_data_list.append(aux_data)

        logit = self.tgt_word_prj(tgt)

        return logit, aux_data_list, script_logits

    def forward_test(self, src):
        """Inference forward pass (greedy autoregressive decoding).

        Args:
            src: [B, N, D] encoder output

        Returns:
            logits: [B, T, vocab_size-2] prediction probabilities
        """
        bs = src.shape[0]
        memory = self._encode_memory(src)

        dec_seq = torch.full(
            (bs, self.max_len + 1), self.ignore_index,
            dtype=torch.int64, device=src.device,
        )
        dec_seq[:, 0] = self.bos

        logits = []
        self.attn_maps = []

        for len_dec_seq in range(0, self.max_len):
            dec_seq_embed = self.embedding(dec_seq[:, :len_dec_seq + 1])
            dec_seq_embed = self.positional_encoding(dec_seq_embed)
            tgt_mask = self.generate_square_subsequent_mask(
                dec_seq_embed.shape[1], src.device)

            tgt = dec_seq_embed
            for decoder_layer in self.decoder:
                tgt, _ = decoder_layer(tgt, memory, self_mask=tgt_mask)

            # Collect attention maps for visualization
            self.attn_maps.append(
                self.decoder[-1].cross_attn.attn_map[0][:, -1:, :])

            dec_output = tgt[:, -1:, :]
            word_prob = F.softmax(self.tgt_word_prj(dec_output), dim=-1)
            logits.append(word_prob)

            if len_dec_seq < self.max_len:
                dec_seq[:, len_dec_seq + 1] = word_prob.squeeze().argmax(-1)
                if (dec_seq == self.eos).any(dim=-1).all():
                    break

        logits = torch.cat(logits, dim=1)
        return logits

    def forward(self, src, data=None):
        """Main forward pass.

        The data format is extended to support script labels:
            Training data = [labels, lengths] or [labels, lengths, script_ids]
            - labels: [B, max_len+2] token indices
            - lengths: [B] text lengths
            - script_ids: [B] script type indices (optional, for supervision)

        During training, returns a dict containing:
            - 'logit': prediction logits
            - 'aux_data_list': MoE routing info for each layer
            - 'script_logits': script classification logits
            - 'script_targets': ground-truth script labels (if provided)

        During inference, returns prediction probabilities directly.
        """
        if self.training:
            max_len = data[1].max()
            tgt = data[0][:, :2 + max_len]
            logit, aux_data_list, script_logits = self.forward_train(src, tgt)

            result = {
                'logit': logit,
                'aux_data_list': aux_data_list,
                'script_logits': script_logits,
            }
            # Pass through script labels if provided
            if len(data) > 2:
                result['script_targets'] = data[2]

            return result
        else:
            return self.forward_test(src)

    def generate_square_subsequent_mask(self, sz, device):
        """Generate a causal mask for autoregressive decoding."""
        mask = torch.zeros([sz, sz], dtype=torch.float32)
        mask_inf = torch.triu(
            torch.full((sz, sz), dtype=torch.float32, fill_value=-torch.inf),
            diagonal=1,
        )
        mask = mask + mask_inf
        return mask.unsqueeze(0).unsqueeze(0).to(device)

    def get_router_stats(self):
        """Get routing statistics for analysis/visualization.

        Returns a dict with per-layer expert load distribution.
        """
        stats = {}
        for i, layer in enumerate(self.decoder):
            router = layer.moe_ffn.router
            stats[f'layer_{i}_router_weight'] = router.router.weight.data.clone()
        return stats
