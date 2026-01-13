import math
from collections import OrderedDict
from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.checkpoint import checkpoint

from transformers import (
    GenerationMixin,
    MBartConfig,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    CausalLMOutputWithCrossAttentions,
)
from transformers.models.mbart.modeling_mbart import MBartDecoder


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.short = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.short = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels))

    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return self.relu(y + self.short(x))


class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        inv_rms = torch.rsqrt(var + self.eps)
        return x * inv_rms * self.weight


class SwiGLU(nn.Module):

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 bias: bool = True):
        super().__init__()
        self.up = nn.Linear(in_features, hidden_features, bias=bias)
        self.gate = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x) * self.act(self.gate(x))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)


def apply_rope2d(q: torch.Tensor, k: torch.Tensor, cos_sin_cache):
    cos_y, sin_y, cos_x, sin_x = cos_sin_cache
    B, nH, M, dH = q.shape
    half = dH // 2
    cy = cos_y.view(1, 1, M, half)
    sy = sin_y.view(1, 1, M, half)
    cx = cos_x.view(1, 1, M, half)
    sx = sin_x.view(1, 1, M, half)
    qy, qx = q[..., :half], q[..., half:]
    ky, kx = k[..., :half], k[..., half:]
    qy = qy * cy + _rotate_half(qy) * sy
    qx = qx * cx + _rotate_half(qx) * sx
    ky = ky * cy + _rotate_half(ky) * sy
    kx = kx * cx + _rotate_half(kx) * sx
    return torch.cat([qy, qx], dim=-1), torch.cat([ky, kx], dim=-1)


class RoPEMHA(nn.Module):

    def __init__(self,
                 dim: int,
                 num_heads: int,
                 attn_drop: float = 0.1,
                 proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(dim, dim, bias=True)
        self.v_proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, cos_sin_cache):
        B, M, D = x.shape
        H, Hd = self.num_heads, self.head_dim
        assert D == H * Hd, f'D={D}, H*Hd={H * Hd}'
        q = self.q_proj(x).view(B, M, H, Hd).transpose(1, 2).contiguous()
        k = self.k_proj(x).view(B, M, H, Hd).transpose(1, 2).contiguous()
        v = self.v_proj(x).view(B, M, H, Hd).transpose(1, 2).contiguous()
        q, k = apply_rope2d(q, k, cos_sin_cache)
        drop_p = self.attn_drop.p if self.training else 0.0
        ctx = (sdpa_kernel([
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]) if torch.cuda.is_available() else nullcontext())
        with ctx:
            attn = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=drop_p,
                is_causal=False,
                scale=self.scale,
            )
        attn = attn.transpose(1, 2).contiguous().view(B, M, D)
        y = self.out_proj(attn)
        return self.proj_drop(y)


class PreNormDecoderLayer(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int,
                 attn_drop_rate: float = 0.1,
                 ffn_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim, eps=1e-6)
        self.mha = RoPEMHA(hidden_dim,
                           num_heads,
                           attn_drop=attn_drop_rate,
                           proj_drop=attn_drop_rate)
        self.norm2 = RMSNorm(hidden_dim, eps=1e-6)
        inner = max(1, int(hidden_dim * ffn_ratio))
        self.ffn = SwiGLU(hidden_dim, inner)
        self.fc_out = nn.Linear(inner, hidden_dim)
        self.drop = nn.Dropout(attn_drop_rate)

    def forward(self, x: torch.Tensor, cos_sin_cache):
        h = self.norm1(x)
        h = self.mha(h, cos_sin_cache)
        x = x + h
        h2 = self.norm2(x)
        h2 = self.fc_out(self.ffn(h2))
        return x + self.drop(h2)


class CMEREncoder(nn.Module):

    def __init__(self,
                 num_layers: int,
                 num_heads: int,
                 hidden_dim: int,
                 *,
                 down_sample_ratio: int = 16,
                 rope_base: float = 10000.0,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.down_sample_ratio = int(down_sample_ratio)
        self.hidden_dim = int(hidden_dim)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.rope_base = float(rope_base)
        self.head_dim = hidden_dim // num_heads
        channels = [3, 12, 24, 48, 96, 192, 384, 768]
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(channels[0], channels[1], stride=2),
            ResidualBlock(channels[1], channels[2], stride=1),
            ResidualBlock(channels[2], channels[3], stride=2),
            ResidualBlock(channels[3], channels[4], stride=1),
            ResidualBlock(channels[4], channels[5], stride=2),
            ResidualBlock(channels[5],
                          channels[6],
                          stride=2 if down_sample_ratio > 16 else 1),
            ResidualBlock(channels[6], channels[7], stride=2),
        ])
        self.fc = nn.Linear(channels[-1], hidden_dim)
        self.vit = nn.ModuleList([
            PreNormDecoderLayer(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])
        self.rope_cache = OrderedDict()
        self.max_rope_cache = getattr(self, 'max_rope_cache', 32)

    def train(self, mode: bool = True):
        prev = self.training
        super().train(mode)
        if mode != prev:
            self.rope_cache.clear()
        return self

    def eval(self):
        prev = self.training
        super().eval()
        if prev:
            self.rope_cache.clear()
        return self

    def clear_rope_cache(self):
        self.rope_cache.clear()

    def _build_rope2d_cache(self, H: int, W: int, device, dtype):
        H = int(H)
        W = int(W)
        key = (H, W, int(self.head_dim))
        if key in self.rope_cache:
            cos_y_cpu, sin_y_cpu, cos_x_cpu, sin_x_cpu = self.rope_cache[key]
            self.rope_cache.move_to_end(key)
        else:
            head_dim = self.head_dim
            assert head_dim % 4 == 0, '2D RoPE 需要 head_dim 能被 4 整除'
            half = head_dim // 2
            inv_freq = 1.0 / (self.rope_base**(torch.arange(
                0, half, 2, device='cpu', dtype=torch.float32) / half))
            pos_y = torch.arange(H, device='cpu', dtype=torch.float32)
            pos_x = torch.arange(W, device='cpu', dtype=torch.float32)
            freqs_y = torch.einsum('i,j->ij', pos_y, inv_freq)
            freqs_x = torch.einsum('i,j->ij', pos_x, inv_freq)
            cos_y_1d = torch.cos(freqs_y).repeat_interleave(2, dim=-1)
            sin_y_1d = torch.sin(freqs_y).repeat_interleave(2, dim=-1)
            cos_x_1d = torch.cos(freqs_x).repeat_interleave(2, dim=-1)
            sin_x_1d = torch.sin(freqs_x).repeat_interleave(2, dim=-1)
            cos_y = cos_y_1d[:, None, :].expand(H, W,
                                                half).reshape(H * W, half)
            sin_y = sin_y_1d[:, None, :].expand(H, W,
                                                half).reshape(H * W, half)
            cos_x = cos_x_1d[None, :, :].expand(H, W,
                                                half).reshape(H * W, half)
            sin_x = sin_x_1d[None, :, :].expand(H, W,
                                                half).reshape(H * W, half)
            entry = tuple(
                t.to(torch.float16).pin_memory()
                for t in (cos_y, sin_y, cos_x, sin_x))
            self.rope_cache[key] = entry
            while len(self.rope_cache) > int(self.max_rope_cache):
                self.rope_cache.popitem(last=False)
            cos_y_cpu, sin_y_cpu, cos_x_cpu, sin_x_cpu = entry
        cos_y = cos_y_cpu.to(device=device, dtype=dtype, non_blocking=True)
        sin_y = sin_y_cpu.to(device=device, dtype=dtype, non_blocking=True)
        cos_x = cos_x_cpu.to(device=device, dtype=dtype, non_blocking=True)
        sin_x = sin_x_cpu.to(device=device, dtype=dtype, non_blocking=True)
        if self.training and torch.is_grad_enabled():
            cos_y = cos_y.clone()
            sin_y = sin_y.clone()
            cos_x = cos_x.clone()
            sin_x = sin_x.clone()
        return (cos_y, sin_y, cos_x, sin_x)

    def forward(self, pixel_values: torch.Tensor):
        x = pixel_values
        for blk in self.residual_blocks:
            x = blk(x)
        N, C, Hc, Wc = x.shape
        seq = x.flatten(2).transpose(1, 2)
        seq = self.fc(seq)
        cos_sin_cache = self._build_rope2d_cache(Hc, Wc, seq.device, seq.dtype)
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled(
        ):

            def _run_layer(layer, s, cache):
                return layer(s, cache)

            for layer in self.vit:
                seq = checkpoint(_run_layer,
                                 layer,
                                 seq,
                                 cos_sin_cache,
                                 use_reentrant=False)
        else:
            for layer in self.vit:
                seq = layer(seq, cos_sin_cache)
        return seq


class CMERConfig(PretrainedConfig):
    model_type = 'CMER'

    def __init__(self, vision_config=None, decoder_config=None, **kwargs):
        self.vision_config = vision_config if vision_config is not None else {}
        self.decoder_config = decoder_config if decoder_config is not None else {}
        if self.decoder_config:
            for key, value in self.decoder_config.items():
                setattr(self, key, value)
        if hasattr(self, 'decoder_layers'):
            self.num_hidden_layers = self.decoder_layers
        super().__init__(**kwargs, **self.decoder_config)


class CMER(PreTrainedModel, GenerationMixin):
    config_class = CMERConfig
    base_model_prefix = 'cmer'
    main_input_name = 'pixel_values'

    def __init__(self, config: CMERConfig):
        super().__init__(config)
        self.config = config
        decoder_config = MBartConfig(**config.decoder_config)
        self.vision_model = CMEREncoder(**config.vision_config)
        self.llm_model = MBartDecoder(decoder_config)
        self.lm_head = torch.nn.Linear(decoder_config.d_model,
                                       decoder_config.vocab_size,
                                       bias=False)
        setattr(self.config, 'tie_word_embeddings', True)
        self.tie_weights()
        setattr(self.lm_head, '_dynamic_tied_weights_keys', ['weight'])
        setattr(self.llm_model.embed_tokens, '_dynamic_tied_weights_keys',
                ['weight'])

    def set_gradient_checkpointing(self, enable: bool = True):
        self.gradient_checkpointing = bool(enable)
        if hasattr(self.vision_model, 'set_gradient_checkpointing'):
            self.vision_model.gradient_checkpointing = self.gradient_checkpointing
        if hasattr(self.llm_model, 'set_gradient_checkpointing'):
            self.llm_model.gradient_checkpointing = self.gradient_checkpointing
            if enable:
                self.llm_model.config.use_cache = False

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_emb):
        self.lm_head = new_emb

    def state_dict(self, *args, **kwargs):
        sd = super().state_dict(*args, **kwargs)
        if 'llm_model.embed_tokens.weight' not in sd and 'lm_head.weight' in sd:
            sd['llm_model.embed_tokens.weight'] = sd['lm_head.weight']
        elif 'lm_head.weight' not in sd and 'llm_model.embed_tokens.weight' in sd:
            sd['lm_head.weight'] = sd['llm_model.embed_tokens.weight']
        return sd

    def load_state_dict(self, state_dict, strict=True):
        if 'llm_model.embed_tokens.weight' not in state_dict and 'lm_head.weight' in state_dict:
            state_dict['llm_model.embed_tokens.weight'] = state_dict[
                'lm_head.weight']
        if 'lm_head.weight' not in state_dict and 'llm_model.embed_tokens.weight' in state_dict:
            state_dict['lm_head.weight'] = state_dict[
                'llm_model.embed_tokens.weight']
        out = super().load_state_dict(state_dict, strict=False)
        self.tie_weights()
        return out

    def get_input_embeddings(self):
        return self.llm_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm_model.set_input_embeddings(value)

    def get_decoder(self):
        return self.llm_model.get_decoder()

    def _swin_stride_and_winsize(self):
        cfg = self.vision_model.config
        patch = int(getattr(cfg, 'patch_size', 4))
        depths = getattr(cfg, 'depths', [2, 2, 6, 2])
        stride = patch * (2**(len(depths) - 1))
        wsize = int(getattr(cfg, 'window_size', 7))
        return stride, wsize

    def _ensure_swin_safe(self, pixel_values: torch.Tensor) -> torch.Tensor:
        stride, wsize = self._swin_stride_and_winsize()
        B, C, H, W = pixel_values.shape
        need_min = wsize * stride
        if min(H, W) < need_min:
            s = need_min / float(min(H, W))
            new_h = math.ceil(H * s / stride) * stride
            new_w = math.ceil(W * s / stride) * stride
            pixel_values = F.interpolate(pixel_values,
                                         size=(new_h, new_w),
                                         mode='bilinear',
                                         align_corners=False)
            H, W = new_h, new_w
        new_h = math.ceil(H / stride) * stride
        new_w = math.ceil(W / stride) * stride
        if (new_h, new_w) != (H, W):
            pixel_values = F.interpolate(pixel_values,
                                         size=(new_h, new_w),
                                         mode='bilinear',
                                         align_corners=False)
        return pixel_values

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[BaseModelOutput] = None,
        past_key_values: Optional[tuple] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithCrossAttentions:

        # 1. 兼容性处理：如果 Trainer 传入的是 'image' 而不是 'pixel_values'
        if pixel_values is None and 'image' in kwargs:
            pixel_values = kwargs.pop('image')

        # 2. 兼容性处理：如果 Trainer 传入的是 'label' 而不是 'labels'
        if labels is None and 'label' in kwargs:
            labels = kwargs.pop('label')

        # 3. Encoder Forward
        if encoder_outputs is None:
            if pixel_values is None:
                raise ValueError(
                    '`pixel_values` must be provided when `encoder_outputs` is not.'
                )
            # pixel_values = self._ensure_swin_safe(pixel_values) # 如果需要 Swin 对齐，取消注释
            encoder_outputs = self.vision_model(pixel_values)

        # 4. 自动生成 decoder_input_ids (Teacher Forcing)
        # 如果没有传 decoder_input_ids，但传了 labels，则使用 labels 作为输入
        # 注意：Processor 已经加了 BOS/EOS，labels 格式通常为 [BOS, token1, token2, EOS]
        # 输入给 Decoder 的应该是 [BOS, token1, token2, EOS]
        # 计算 Loss 时，logits 会取 [:-1]，labels 会取 [1:]，从而实现预测下一个 token
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = labels.clone()
            # 将 -100 (ignore_index) 替换为 pad_token_id，防止 embedding 越界
            pad_token_id = self.config.decoder_config.get(
                'pad_token_id',
                self.config.decoder_config.get('eos_token_id',
                                               1))  # 默认 fallback
            decoder_input_ids.masked_fill_(decoder_input_ids == -100,
                                           pad_token_id)

        # 5. Decoder Forward
        # 此时 decoder_input_ids 应该已经有值了，不会再报 ValueError
        decoder_outputs = self.llm_model(
            input_ids=decoder_input_ids,
            inputs_embeds=None,  # <--- 强制为 None，解决报错
            encoder_hidden_states=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=False,
            return_dict=True,
            # 注意：不要在这里传入 **kwargs，因为 kwargs 可能包含 'decoder_inputs_embeds' 等导致冲突的键
        )

        logits = self.lm_head(decoder_outputs.last_hidden_state)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            eps = getattr(self.config, 'label_smoothing', 0.1)
            loss = F.cross_entropy(
                shift_logits.view(-1,
                                  self.config.decoder_config['vocab_size']),
                shift_labels.view(-1),
                ignore_index=-100,
                label_smoothing=eps,
            )

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=None
            if self.training else decoder_outputs.past_key_values,
            hidden_states=None,
            attentions=None,
            cross_attentions=None,
        )

    def prepare_inputs_for_generation(self,
                                      input_ids,
                                      past_key_values=None,
                                      **kwargs):
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
        return {
            'decoder_input_ids': input_ids,
            'past_key_values': past_key_values,
            'encoder_outputs': kwargs.get('encoder_outputs'),
            'attention_mask': kwargs.get('attention_mask'),
        }

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 256,
        do_sample: bool = False,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        bos_token_id: Optional[int] = 2,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        return_only_new_tokens: bool = True,
        num_beams: int = 1,
        **kwargs,
    ):
        if num_beams != 1:
            raise NotImplementedError(
                '当前极简 generate 未实现 beam search（num_beams>1）。')
        device = pixel_values.device if pixel_values is not None else next(
            self.parameters()).device
        encoder_outputs = kwargs.get('encoder_outputs', None)
        if encoder_outputs is None:
            if pixel_values is None:
                raise ValueError(
                    '`pixel_values` is required if `encoder_outputs` is not provided.'
                )
            enc = self.vision_model(pixel_values)
            encoder_hidden_states = enc
        else:
            if isinstance(encoder_outputs, (tuple, list)):
                encoder_hidden_states = encoder_outputs[0]
            elif hasattr(encoder_outputs, 'last_hidden_state'):
                encoder_hidden_states = encoder_outputs.last_hidden_state
            elif isinstance(encoder_outputs,
                            dict) and 'last_hidden_state' in encoder_outputs:
                encoder_hidden_states = encoder_outputs['last_hidden_state']
            else:
                raise ValueError(
                    '`encoder_outputs` 格式不正确，缺少 last_hidden_state。')
            encoder_hidden_states = encoder_hidden_states.to(device)
        batch_size = encoder_hidden_states.size(0)

        bos_id = bos_token_id

        if eos_token_id is None:
            eos_token_id = kwargs.get('eos_token_id', None)
            if eos_token_id is None:
                eos_token_id = -1
        if pad_token_id is None:
            pad_token_id = kwargs.get('pad_token_id', None)
            if pad_token_id is None:
                pad_token_id = bos_id
        if decoder_input_ids is None:
            input_ids = torch.full((batch_size, 1),
                                   bos_id,
                                   dtype=torch.long,
                                   device=device)
        else:
            input_ids = decoder_input_ids.to(device)
        self.llm_model.config.use_cache = True
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        def _top_k_top_p_filtering(logits,
                                   top_k=0,
                                   top_p=1.0,
                                   min_tokens_to_keep=1):
            top_k = min(max(top_k, 0), logits.size(-1))
            if top_k > 0:
                kth_vals, _ = torch.topk(logits, top_k)
                min_thresh = kth_vals[..., -1, None]
                logits = torch.where(logits < min_thresh,
                                     torch.full_like(logits, float('-inf')),
                                     logits)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits,
                                                           descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = probs.cumsum(dim=-1)
                sorted_mask = cumulative_probs > top_p
                if min_tokens_to_keep > 0:
                    sorted_mask[..., :min_tokens_to_keep] = 0
                sorted_logits = sorted_logits.masked_fill(
                    sorted_mask, float('-inf'))
                logits = torch.zeros_like(logits).scatter(dim=-1,
                                                          index=sorted_indices,
                                                          src=sorted_logits)
            return logits

        for _ in range(max_new_tokens):
            dec_in = input_ids[:, -1:]
            dec_out = self.llm_model(
                input_ids=dec_in,
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = dec_out.past_key_values
            hidden = dec_out.last_hidden_state
            logits = self.lm_head(hidden[:, -1, :])
            if do_sample:
                logits = logits / max(temperature, 1e-6)
                logits = _top_k_top_p_filtering(logits,
                                                top_k=top_k,
                                                top_p=top_p,
                                                min_tokens_to_keep=1)
                probs = torch.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs,
                                                num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(logits, dim=-1)
            next_tokens = torch.where(
                finished, torch.full_like(next_tokens, pad_token_id),
                next_tokens)
            input_ids = torch.cat(
                [input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            if eos_token_id >= 0:
                finished = finished | (next_tokens == eos_token_id)
                if torch.all(finished):
                    break
        if return_only_new_tokens:
            if decoder_input_ids is None:
                return input_ids[:, 1:]
            else:
                return input_ids[:, decoder_input_ids.size(1):]
        else:
            return input_ids


def build_model_cmer(config):
    backbone_config = config.get('Backbone', {})

    vision_cfg = backbone_config.get('vision_config', {})
    decoder_cfg = backbone_config.get('decoder_config', {})

    cmer_config = CMERConfig(vision_config=vision_cfg,
                             decoder_config=decoder_cfg)

    model = CMER(cmer_config)

    return model
