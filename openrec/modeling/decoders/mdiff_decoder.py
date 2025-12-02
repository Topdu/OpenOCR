import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from openrec.modeling.common import Mlp
from openrec.modeling.decoders.nrtr_decoder import PositionalEncoding, Embeddings, MultiheadAttention


class MDiffDecoder(nn.Module):
    """A transformer model. User is able to modify the attributes as needed.
    The architechture is based on the paper "Attention Is All You Need". Ashish
    Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N
    Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you
    need. In Advances in Neural Information Processing Systems, pages
    6000-6010.

    Args:
        d_model: the number of expected features in the encoder/decoder inputs (default=512).
        nhead: the number of heads in the multiheadattention models (default=8).
        num_encoder_layers: the number of sub-encoder-layers in the encoder (default=6).
        num_decoder_layers: the number of sub-decoder-layers in the decoder (default=6).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        custom_encoder: custom encoder (default=None).
        custom_decoder: custom decoder (default=None).
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 nhead=None,
                 num_decoder_layers=6,
                 max_len=25,
                 attention_dropout_rate=0.0,
                 residual_dropout_rate=0.1,
                 scale_embedding=True,
                 parallel_decoding=False,
                 autoregressive_decoding=False,
                 sampler_step=5,
                 low_confidence_decoding=False,
                 random_mask_decoding=False,
                 semi_autoregressive_decoding=False,
                 cloze_mask_decoding=False,
                 rec_loss_weight=1.0,
                 reflect_loss_weight=1.0,
                 sample_k=0,
                 temperature=1.0):
        super(MDiffDecoder, self).__init__()
        self.out_channels = out_channels
        self.ignore_index = out_channels - 1
        self.mask_token_id = out_channels - 2
        self.eos = 0
        self.max_len = max_len
        d_model = in_channels
        dim_feedforward = d_model * 4
        self.pd = parallel_decoding
        self.ar = autoregressive_decoding
        self.sampler_step = sampler_step
        self.lc = low_confidence_decoding
        self.rm = random_mask_decoding
        self.semiar = semi_autoregressive_decoding
        self.cm = cloze_mask_decoding
        self.rec_loss_weight = rec_loss_weight
        self.reflect_loss_weight = reflect_loss_weight
        self.temperature = temperature
        self.sample_k = sample_k
        nhead = nhead if nhead is not None else d_model // 32
        self.embedding = Embeddings(
            d_model=d_model,
            vocab=self.out_channels,
            padding_idx=0,
            scale_embedding=scale_embedding,
        )
        self.pos_embed = nn.Parameter(torch.zeros(
            [1, self.max_len + 1, d_model], dtype=torch.float32),
                                      requires_grad=True)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.positional_encoding = PositionalEncoding(
            dropout=residual_dropout_rate, dim=d_model)

        self.decoder = nn.ModuleList([
            TransformerBlock(
                d_model,
                nhead,
                dim_feedforward,
                attention_dropout_rate,
                residual_dropout_rate,
                with_self_attn=True,
                with_cross_attn=True,
            ) for i in range(num_decoder_layers)
        ])

        self.num_decoder_layers = num_decoder_layers

        self.d_model = d_model
        self.nhead = nhead
        self.tgt_word_prj = nn.Linear(d_model,
                                      self.out_channels - 2,
                                      bias=False)
        w0 = np.random.normal(0.0, d_model**-0.5,
                              (d_model, self.out_channels - 2)).astype(
                                  np.float32)
        self.tgt_word_prj.weight.data = torch.from_numpy(w0.transpose())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_train(self, memory, data=None):
        labels, reflect_ids, noisy_batch, masked_indices, p_mask, length = data
        p_mask = p_mask[:, None].repeat(1, labels.shape[1])
        noisy_data_length = length + 1
        noisy_data_length = noisy_data_length[:,
                                              None].repeat(1, labels.shape[1])

        tgts = self.embedding(noisy_batch)
        tgts = self.positional_encoding(tgts) + self.pos_embed

        for decoder_layer in self.decoder:
            tgts = decoder_layer(tgts, memory, self_mask=None)
        logits = self.tgt_word_prj(tgts)
        token_loss = F.cross_entropy(
            logits[masked_indices],
            labels[masked_indices],
            reduction='none',
            ignore_index=self.ignore_index) / p_mask[masked_indices]
        loss = torch.sum(
            token_loss / noisy_data_length[masked_indices]) / labels.shape[0]

        if reflect_ids is not None:
            reflect_tgts = self.embedding(reflect_ids)
            reflect_tgts = self.positional_encoding(
                reflect_tgts) + self.pos_embed
            for decoder_layer in self.decoder:
                reflect_tgts = decoder_layer(reflect_tgts,
                                             memory,
                                             self_mask=None)
            reflect_logits = self.tgt_word_prj(reflect_tgts)
            reflect_loss = F.cross_entropy(reflect_logits.flatten(0, 1),
                                           labels.flatten(0, 1),
                                           reduction='mean',
                                           ignore_index=self.ignore_index)
            loss = self.rec_loss_weight * loss + self.reflect_loss_weight * reflect_loss

        return loss

    def forward_train_all(self, memory, data=None):

        labels, reflect_ids_all, noisy_batch_all, masked_indices_all, p_mask_all, length = data
        bs, L = labels.shape
        tgts = self.embedding(noisy_batch_all.flatten(0, 1))
        tgts = self.positional_encoding(tgts) + self.pos_embed
        tgts = tgts.reshape(bs, self.sample_k, L, -1)

        for decoder_layer in self.decoder:
            tgts = decoder_layer(tgts,
                                 memory,
                                 self_mask=None,
                                 sample_k=self.sample_k)
        logits_all = self.tgt_word_prj(tgts)  # bs, sample_k, L, c_num

        reflect_tgts = self.embedding(reflect_ids_all.flatten(0, 1))
        reflect_tgts = self.positional_encoding(reflect_tgts) + self.pos_embed
        reflect_tgts = reflect_tgts.reshape(bs, self.sample_k, L, -1)

        for decoder_layer in self.decoder:
            reflect_tgts = decoder_layer(reflect_tgts,
                                         memory,
                                         self_mask=None,
                                         sample_k=self.sample_k)
        reflect_logits_all = self.tgt_word_prj(reflect_tgts)

        loss = []
        for i in range(self.sample_k):
            p_mask = p_mask_all[:, i]
            masked_indices = masked_indices_all[:, i]
            logits = logits_all[:, i]

            p_mask = p_mask[:, None].repeat(1, labels.shape[1])
            noisy_data_length = length + 1
            noisy_data_length = noisy_data_length[:, None].repeat(
                1, labels.shape[1])
            token_loss = F.cross_entropy(
                logits[masked_indices],
                labels[masked_indices],
                reduction='none',
                ignore_index=self.ignore_index) / p_mask[masked_indices]
            denoise_loss_i = torch.sum(
                token_loss /
                noisy_data_length[masked_indices]) / labels.shape[0]

            reflect_logits = reflect_logits_all[:, i]
            reflect_loss_i = F.cross_entropy(reflect_logits.flatten(0, 1),
                                             labels.flatten(0, 1),
                                             reduction='mean',
                                             ignore_index=self.ignore_index)
            loss_i = self.rec_loss_weight * denoise_loss_i + self.reflect_loss_weight * reflect_loss_i
            loss.append(loss_i)

        return sum(loss) / len(loss)

    def forward(self, src, data=None):
        """Take in and process masked source/target sequences.
        Args:
            src: the sequence to the encoder (required).
            tgt: the sequence to the decoder (required).
        Shape:
            - src: :math:`(B, sN, C)`.
            - tgt: :math:`(B, tN, C)`.
        Examples:
            >>> output = transformer_model(src, tgt)
        """

        if self.training:
            if self.sample_k > 0:
                res = self.forward_train_all(src, data)
            else:
                res = self.forward_train(src, data)
        else:
            if self.pd:
                res = self.forward_parallel_decoding(src)
            elif self.ar:
                res = self.forward_autoregressive_decoding(src)
            elif self.lc:
                res = self.forward_low_confidence_decoding(src)
            elif self.rm:
                res = self.forward_random_mask_decoding(src)
            elif self.semiar:
                res = self.forward_semi_autoregressive_decoding(src)
            elif self.cm:
                res = self.forward_cloze_mask_decoding(src)
            else:
                res = self.forward_parallel_decoding(src)

        return res

    def forward_decoding(self, src, tgts, step_i=0):

        tgts = self.embedding(tgts)
        tgts = self.positional_encoding(tgts) + self.pos_embed
        for decoder_layer in self.decoder:
            tgts = decoder_layer(tgts, src, self_mask=None)

        return tgts

    def forward_reflect(self, src, pred_indexs, step_i=0):
        """Reflect decoding."""

        # reflect
        masked_indices_eos = self.get_masked_indice_after_eos(
            pred_indexs
        )  # [bs, max_len + 1] bool tensor False False(eos) True True ..
        pred_indexs[
            masked_indices_eos] = self.mask_token_id  # 保留eos之后的token为mask token

        reflect_tgts = self.forward_decoding(src, pred_indexs, step_i=step_i)
        logits_reflect = F.softmax(self.tgt_word_prj(reflect_tgts), -1)

        return logits_reflect

    def forward_parallel_decoding(self, src):
        bs = src.shape[0]
        noisy_batch = torch.full((bs, self.max_len + 1),
                                 self.mask_token_id,
                                 dtype=torch.int64,
                                 device=src.get_device())
        tgts = self.forward_decoding(src, noisy_batch)
        logits = F.softmax(self.tgt_word_prj(tgts), -1)
        return logits

    def get_masked_indice_after_eos(self, noisy_batch):
        """Get the indices of the masked tokens after the first EOS token."""
        # noisy_batch: [batch_size, max_len + 1]
        eos_mask = noisy_batch == self.eos  # [batch_size, seq_len]

        # 找到每行第一个eos的位置
        eos_indices = eos_mask.float().argmax(dim=1)  # [batch_size]

        # 如果没有eos，argmax会返回0，但我们不想在这些地方mask，需要过滤
        eos_exists = eos_mask.any(dim=1)  # [batch_size]

        batch_size, seq_len = noisy_batch.shape
        arange = torch.arange(seq_len,
                              device=noisy_batch.device).unsqueeze(0).expand(
                                  batch_size, -1)  # [batch_size, seq_len]

        # 创建掩码：只对eos之后的token设为True
        masked_indices = arange > eos_indices.unsqueeze(1)
        masked_indices = masked_indices | ~eos_exists.unsqueeze(1)

        return masked_indices

    def forward_low_confidence_decoding(self, src):
        bs = src.shape[0]
        noisy_batch = torch.full((bs, self.max_len + 1),
                                 self.mask_token_id,
                                 dtype=torch.int64,
                                 device=src.get_device())
        masked_indices_pre = torch.full((bs, self.max_len + 1),
                                        True,
                                        dtype=torch.bool,
                                        device=src.get_device())
        flag_exit = False
        for step_i in range(self.sampler_step):

            tgts = self.forward_decoding(src, noisy_batch, step_i=step_i)
            pred_step = self.tgt_word_prj(tgts)
            pred_step = F.softmax(pred_step, -1)
            if step_i == 0:
                logits = pred_step.clone()
            logits[masked_indices_pre] = pred_step[masked_indices_pre]
            pred_step_prob, pred_step_index = torch.max(
                pred_step, dim=-1)  # [bs, max_len + 1], [bs, max_len + 1]
            masked_indices_eos = self.get_masked_indice_after_eos(
                pred_step_index
            )  # [bs, max_len + 1] bool tensor False False(eos) True True ..

            # 仅计算mask token位置以及eos之前token的平均概率
            valid_indices = masked_indices_pre & ~masked_indices_eos
            pred_step_prob = pred_step_prob * valid_indices.float()
            pred_step_prob_avg = pred_step_prob.sum(
                dim=1, keepdim=True) / valid_indices.sum(
                    dim=1, keepdim=True)  # [bs, 1]

            # 高于平均置信度的token
            top_confidence_mask = pred_step_prob > pred_step_prob_avg
            top_confidence_mask = top_confidence_mask & valid_indices
            noisy_batch[top_confidence_mask] = pred_step_index[
                top_confidence_mask]
            # 低置信度的token或者eos之后的token均保留为 self.mask_token_id， 其他则替换为 pred_step_index
            masked_indices_pre = noisy_batch == self.mask_token_id
            masked_indices_vaild = masked_indices_pre & ~masked_indices_eos
            if flag_exit:
                # 如果已经满足退出条件，直接返回
                break
            if (masked_indices_vaild.sum(dim=-1) <= 1).all():
                # 如果每个batch中只有一个或者0个token被mask，说明下次已经没有足够的token可以被mask了，再进行一次就结束
                flag_exit = True

        return logits

    def forward_random_mask_decoding(self, src):
        bs = src.shape[0]
        noisy_batch = torch.full((bs, self.max_len + 1),
                                 self.mask_token_id,
                                 dtype=torch.int64,
                                 device=src.get_device())
        masked_indices_pre = torch.full((bs, self.max_len + 1),
                                        True,
                                        dtype=torch.bool,
                                        device=src.get_device())
        flag_exit = False
        for step_i in range(self.sampler_step):

            tgts = self.forward_decoding(src, noisy_batch, step_i=step_i)

            pred_step = self.tgt_word_prj(tgts)
            pred_step = F.softmax(pred_step, -1)
            if step_i == 0:
                logits = pred_step.clone()
            else:
                logits[masked_indices_pre] = pred_step[masked_indices_pre]
            pred_step_prob, pred_step_index = torch.max(
                pred_step, dim=-1)  # [bs, max_len + 1], [bs, max_len + 1]
            masked_indices_eos = self.get_masked_indice_after_eos(
                pred_step_index)  # [bs, max_len + 1] bool tensor

            # 采用mask token位置以及eos之前token作为可用token
            valid_indices = masked_indices_pre & ~masked_indices_eos
            # 在这些可用token中随机选择一些进行mask
            rand_mask_prob = torch.rand((bs, self.max_len + 1),
                                        device=src.get_device())
            # rand_mask_prob = rand_mask_prob * valid_indices.float()
            random_res = rand_mask_prob > 0.5  # 50%的概率进行mask
            # 仅保留mask token位置以及eos之前token的高置信度token
            random_res = random_res & valid_indices
            # random_mask = random_mask & masked_indices_pre
            noisy_batch[random_res] = pred_step_index[random_res]
            # 随机mask token或者eos之后的token均保留为 self.mask_token_id， 其他则替换为 pred_step_index
            masked_indices_pre = noisy_batch == self.mask_token_id
            masked_indices_vaild = masked_indices_pre & ~masked_indices_eos
            if flag_exit:
                # 如果已经满足退出条件，直接返回
                break
            if (masked_indices_vaild.sum(dim=-1) <= 1).all():
                # 如果每个batch中只有一个或者0个token被mask，说明下次已经没有足够的token可以被mask了，再进行一次就结束
                flag_exit = True

        return logits

    def forward_semi_autoregressive_decoding(self, src):
        bs = src.shape[0]
        noisy_batch = torch.full((bs, self.max_len + 1),
                                 self.mask_token_id,
                                 dtype=torch.int64,
                                 device=src.get_device())
        block_size = (self.max_len + 1) // self.sampler_step
        masked_indices_pre = torch.full((bs, self.max_len + 1),
                                        True,
                                        dtype=torch.bool,
                                        device=src.get_device())
        flag_exit = False
        for step_i in range(self.sampler_step):

            tgts = self.forward_decoding(src, noisy_batch, step_i=step_i)

            pred_step = self.tgt_word_prj(tgts)

            pred_step = pred_step / self.temperature
            pred_step = F.softmax(pred_step, -1)
            if step_i == 0:
                logits = pred_step.clone()
            else:
                logits[masked_indices_pre] = pred_step[masked_indices_pre]
            pred_step_prob, pred_step_index = torch.max(
                pred_step, dim=-1)  # [bs, max_len + 1], [bs, max_len + 1]
            masked_indices_eos = self.get_masked_indice_after_eos(
                pred_step_index
            )  # [bs, max_len + 1] bool tensor False False(eos) True True ..

            block_vaild_indices = torch.full((bs, self.max_len + 1),
                                             False,
                                             dtype=torch.bool,
                                             device=src.get_device())

            if step_i <= 2:
                if self.sampler_step > 2:
                    block_vaild_indices[:, :block_size * (step_i + 1)] = True
                else:
                    block_vaild_indices = ~block_vaild_indices
            elif step_i >= self.sampler_step - 2:
                block_vaild_indices[:, block_size * (step_i - 1):] = True
            else:
                block_vaild_indices[:, block_size * (step_i - 1):block_size *
                                    (step_i + 1)] = True

            # 仅计算mask token位置, eos之前token以及当前block中token的平均概率
            valid_indices = masked_indices_pre & ~masked_indices_eos & block_vaild_indices
            pred_step_prob = pred_step_prob * valid_indices.float()
            pred_step_prob_avg = pred_step_prob.sum(
                dim=1, keepdim=True) / valid_indices.sum(
                    dim=1, keepdim=True)  # [bs, 1]

            # 高于平均置信度的token
            top_confidence_mask = pred_step_prob > pred_step_prob_avg
            top_confidence_mask = top_confidence_mask & valid_indices

            noisy_batch[top_confidence_mask] = pred_step_index[
                top_confidence_mask]

            # 低置信度的token或者eos之后的token均保留为 self.mask_token_id， 其他则替换为 pred_step_index
            masked_indices_pre = noisy_batch == self.mask_token_id
            masked_indices_vaild = masked_indices_pre & ~masked_indices_eos
            if flag_exit:
                # 如果已经满足退出条件，直接返回
                break
            if (masked_indices_vaild.sum(dim=-1) <= 1).all():
                # 如果每个batch中只有一个或者0个token被mask，说明下次已经没有足够的token可以被mask了，再进行一次就结束
                flag_exit = True

        return logits

    def forward_autoregressive_decoding(self, src):
        bs = src.shape[0]
        noisy_batch = torch.full((bs, self.max_len + 1),
                                 self.mask_token_id,
                                 dtype=torch.int64,
                                 device=src.get_device())
        logits = []
        for step_i in range(self.max_len + 1):

            tgts = self.forward_decoding(src, noisy_batch, step_i=step_i)

            pred_step = self.tgt_word_prj(tgts[:, step_i:step_i + 1, :])
            pred_step = F.softmax(pred_step, -1)
            logits.append(pred_step)
            pred_step = torch.argmax(pred_step, dim=-1)
            noisy_batch[:, step_i] = pred_step[:, 0]
            if (noisy_batch == self.eos).any(dim=-1).all():
                break
        logits = torch.cat(logits, dim=1)
        return logits

    def forward_cloze_mask_decoding(self, src, noisy_batch=None):
        """Cloze Mask Decoding."""
        bs = src.shape[0]
        if noisy_batch is None:
            noisy_batch = torch.full((bs, self.max_len + 1),
                                     self.mask_token_id,
                                     dtype=torch.int64,
                                     device=src.get_device())
            tgts = self.forward_decoding(src, noisy_batch)
            pred_step = self.tgt_word_prj(tgts)
            pred_step = F.softmax(pred_step, -1)
            noisy_batch = torch.argmax(pred_step, dim=-1)
            masked_indices_eos = self.get_masked_indice_after_eos(
                noisy_batch)  # [bs, max_len + 1] bool tensor
            noisy_batch[
                masked_indices_eos] = self.mask_token_id  # 保留eos之后的token为mask token

        logits = torch.rand((bs, self.max_len + 1, self.out_channels - 2),
                            dtype=torch.float32,
                            device=src.get_device())
        for step_i in range(self.max_len + 1):
            noisy_batch[:, step_i] = self.mask_token_id

            tgts = self.forward_decoding(src, noisy_batch, step_i=step_i)

            pred_step = self.tgt_word_prj(tgts[:, step_i:step_i + 1, :])
            pred_step = F.softmax(pred_step, -1)
            logits[:, step_i:step_i + 1, :] = pred_step
            pred_step = torch.argmax(pred_step, dim=-1)
            noisy_batch[:, step_i] = pred_step[:, 0]
            if (torch.argmax(logits, dim=-1) == self.eos).any(dim=-1).all():
                break
        return logits


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        attention_dropout_rate=0.0,
        residual_dropout_rate=0.1,
        with_self_attn=True,
        with_cross_attn=False,
        epsilon=1e-5,
    ):
        super(TransformerBlock, self).__init__()
        self.with_self_attn = with_self_attn
        if with_self_attn:
            self.self_attn = MultiheadAttention(d_model,
                                                nhead,
                                                dropout=attention_dropout_rate,
                                                self_attn=with_self_attn)
            self.norm1 = nn.LayerNorm(d_model, eps=epsilon)
            self.dropout1 = nn.Dropout(residual_dropout_rate)
        self.with_cross_attn = with_cross_attn
        if with_cross_attn:
            self.cross_attn = MultiheadAttention(
                d_model, nhead, dropout=attention_dropout_rate
            )  # for self_attn of encoder or cross_attn of decoder
            self.norm2 = nn.LayerNorm(d_model, eps=epsilon)
            self.dropout2 = nn.Dropout(residual_dropout_rate)

        self.mlp = Mlp(
            in_features=d_model,
            hidden_features=dim_feedforward,
            act_layer=nn.ReLU,
            drop=residual_dropout_rate,
        )

        self.norm3 = nn.LayerNorm(d_model, eps=epsilon)

        self.dropout3 = nn.Dropout(residual_dropout_rate)

    def forward(self,
                tgt,
                memory=None,
                self_mask=None,
                cross_mask=None,
                sample_k=0):

        if self.with_self_attn:
            if sample_k > 0:
                bs, _, L, Dim = tgt.shape
                tgt = tgt.flatten(0, 1)
            tgt1 = self.self_attn(tgt, attn_mask=self_mask)
            tgt = self.norm1(tgt + self.dropout1(tgt1))

        if self.with_cross_attn:
            if sample_k > 0:
                tgt = tgt.reshape(bs, sample_k, L, Dim).flatten(1, 2)
            tgt2 = self.cross_attn(tgt, key=memory, attn_mask=cross_mask)
            tgt = self.norm2(tgt + self.dropout2(tgt2))
        tgt = self.norm3(tgt + self.dropout3(self.mlp(tgt)))

        if sample_k > 0:
            tgt = tgt.reshape(bs, sample_k, L, Dim)

        return tgt
