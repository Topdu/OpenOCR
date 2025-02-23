import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import ones_, trunc_normal_, zeros_

from openrec.modeling.common import DropPath, Identity, Mlp
from openrec.modeling.decoders.nrtr_decoder import Embeddings


class CrossAttention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, kv, key_mask=None):
        N, C = kv.shape[1:]
        QN = q.shape[1]
        q = self.q(q).reshape([-1, QN, self.num_heads,
                               C // self.num_heads]).transpose(1, 2)
        q = q * self.scale
        k, v = self.kv(kv).reshape(
            [-1, N, 2, self.num_heads,
             C // self.num_heads]).permute(2, 0, 3, 1, 4)

        attn = q.matmul(k.transpose(2, 3))

        if key_mask is not None:
            attn = attn + key_mask.unsqueeze(1)

        attn = F.softmax(attn, -1)
        if not self.training:
            self.attn_map = attn
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).transpose(1, 2).reshape((-1, QN, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class EdgeDecoderLayer(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=[0.0, 0.0],
        act_layer=nn.GELU,
        norm_layer='nn.LayerNorm',
        epsilon=1e-6,
    ):
        super().__init__()

        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path1 = DropPath(
            drop_path[0]) if drop_path[0] > 0.0 else Identity()
        self.norm1 = eval(norm_layer)(dim, eps=epsilon)
        self.norm2 = eval(norm_layer)(dim, eps=epsilon)

        # self.c = nn.Linear(dim, dim*2)
        self.p = nn.Linear(dim, dim)
        self.cv = nn.Linear(dim, dim)
        self.pv = nn.Linear(dim, dim)

        self.dim = dim
        self.num_heads = num_heads
        self.p_proj = nn.Linear(dim, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, p, cv, pv):
        pN = p.shape[1]
        vN = cv.shape[1]
        p_shortcut = p

        p1 = self.p(p).reshape(
            [-1, pN, self.num_heads,
             self.dim // self.num_heads]).transpose(1, 2)
        cv1 = self.cv(cv).reshape(
            [-1, vN, self.num_heads,
             self.dim // self.num_heads]).transpose(1, 2)
        pv1 = self.pv(pv).reshape(
            [-1, vN, self.num_heads,
             self.dim // self.num_heads]).transpose(1, 2)

        edge = F.softmax(p1.matmul(pv1.transpose(2, 3)), -1)  # B h N N

        p_c = (edge @ cv1).transpose(1, 2).reshape((-1, pN, self.dim))

        x1 = self.norm1(p_shortcut + self.drop_path1(self.p_proj(p_c)))

        x = self.norm2(x1 + self.drop_path1(self.mlp(x1)))
        return x


class DecoderLayer(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer='nn.LayerNorm',
        epsilon=1e-6,
    ):
        super().__init__()
        self.norm1 = eval(norm_layer)(dim, eps=epsilon)
        self.normkv = eval(norm_layer)(dim, eps=epsilon)

        self.mixer = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()

        self.norm2 = eval(norm_layer)(dim, eps=epsilon)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, q, kv, key_mask=None):
        x1 = q + self.drop_path(
            self.mixer(self.norm1(q), self.normkv(kv), key_mask))
        x = x1 + self.drop_path(self.mlp(self.norm2(x1)))
        return x


class CMFFLayer(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        epsilon=1e-6,
    ):
        super().__init__()
        self.normq1 = nn.LayerNorm(dim, eps=epsilon)
        self.normkv1 = nn.LayerNorm(dim, eps=epsilon)
        self.images_to_question_cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.normq2 = nn.LayerNorm(dim, eps=epsilon)
        self.normkv2 = nn.LayerNorm(dim, eps=epsilon)
        self.question_to_images_cross_attn = CrossAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.normmlp = nn.LayerNorm(dim, eps=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, question_f, prompt_f, visual_f, mask=None):

        query_add = torch.concat([question_f, prompt_f, visual_f], 1)

        query_add = query_add + self.drop_path(
            self.images_to_question_cross_attn(self.normq1(query_add),
                                               self.normkv1(prompt_f), mask))
        query_add = query_add + self.drop_path(
            self.question_to_images_cross_attn(
                self.normq2(query_add),
                self.normkv2(query_add[:, -visual_f.shape[1]:, :])))
        query_updated = query_add + self.drop_path(
            self.mlp(self.normmlp(query_add)))

        question_f_updated = query_updated[:, :question_f.shape[1], :]
        prompt_f_updated = query_updated[:, question_f.
                                         shape[1]:-visual_f.shape[1], :]
        visual_f_updated = query_updated[:, -visual_f.shape[1]:, :]

        return question_f_updated, prompt_f_updated, visual_f_updated


class IGTRDecoder(nn.Module):
    """
    IGTRDecoder is a neural network module designed for decoding tasks in OCR (Optical Character Recognition) systems.
    It utilizes a combination of embedding layers, multi-head attention layers, and linear layers to process input sequences
    and generate output sequences.

    Args:
        in_channels (int): Number of input channels.
        dim (int): Dimension of the model.
        out_channels (int): Number of output channels.
        num_layer (int, optional): Number of layers in the decoder. Default is 2.
        drop_path_rate (float, optional): Drop path rate for stochastic depth. Default is 0.1.
        max_len (int, optional): Maximum length of the sequence. Default is 25.
        vis_seq (int, optional): Length of the visual sequence. Default is 50.
        ch (bool, optional): Flag for character embedding. Default is False.
        ar (bool, optional): Flag for autoregressive decoding. Default is False.
        refine_iter (int, optional): Number of refinement iterations. Default is 0.
        quesall (bool, optional): Flag to use all questions. Default is True.
        next_pred (bool, optional): Flag for next prediction. Default is False.
        ds (bool, optional): Flag for downsampling. Default is False.
        pos2d (bool, optional): Flag for 2D positional embedding. Default is False.
        check_search (bool, optional): Flag for checking search. Default is False.
        max_size (list, optional): Maximum size for 2D positional embedding. Default is [8, 32].
        **kwargs: Additional keyword arguments.

    Methods:
        _init_weights(m): Initializes the weights of the module.
        no_weight_decay(): Returns the parameters that should not have weight decay.
        question_encoder(targets, train_i): Encodes the questions based on the targets and training index.
        forward(x, data=None): Forward pass of the decoder. Calls either forward_train or forward_test based on the mode.
        forward_test(x): Forward pass during testing.
        forward_train(x, targets=None): Forward pass during training.

    Returns:
        Depending on the mode (training or testing), the forward method returns either the loss and logits (during training)
        or the predicted indices and probabilities (during testing).
    """

    def __init__(self,
                 in_channels,
                 dim,
                 out_channels,
                 num_layer=2,
                 drop_path_rate=0.1,
                 max_len=25,
                 vis_seq=50,
                 ch=False,
                 ar=False,
                 refine_iter=0,
                 quesall=True,
                 next_pred=False,
                 ds=False,
                 pos2d=False,
                 check_search=False,
                 max_size=[8, 32],
                 **kwargs):
        super(IGTRDecoder, self).__init__()

        self.out_channels = out_channels
        self.dim = dim
        self.max_len = max_len + 3  # max_len + eos + bos
        self.ch = ch
        self.char_embed = Embeddings(d_model=dim,
                                     vocab=self.out_channels,
                                     scale_embedding=True)
        self.ignore_index = out_channels - 1
        self.ar = ar
        self.refine_iter = refine_iter
        self.bos = self.out_channels - 2
        self.eos = 0
        self.next_pred = next_pred
        self.quesall = quesall
        self.check_search = check_search
        dpr = np.linspace(0, drop_path_rate, num_layer + 2)

        self.cmff_decoder = nn.ModuleList([
            CMFFLayer(dim=dim,
                      num_heads=dim // 32,
                      mlp_ratio=4.0,
                      qkv_bias=True,
                      drop_path=dpr[i]) for i in range(num_layer)
        ])

        self.answer_to_question_layer = DecoderLayer(dim=dim,
                                                     num_heads=dim // 32,
                                                     mlp_ratio=4.0,
                                                     qkv_bias=True,
                                                     drop_path=dpr[-2])
        self.answer_to_image_layer = DecoderLayer(dim=dim,
                                                  num_heads=dim // 32,
                                                  mlp_ratio=4.0,
                                                  qkv_bias=True,
                                                  drop_path=dpr[-1])

        self.char_pos_embed = nn.Parameter(torch.zeros([self.max_len, dim],
                                                       dtype=torch.float32),
                                           requires_grad=True)
        self.appear_num_embed = nn.Parameter(torch.zeros([self.max_len, dim],
                                                         dtype=torch.float32),
                                             requires_grad=True)
        self.ds = ds
        self.pos2d = pos2d
        if not ds:
            self.vis_pos_embed = nn.Parameter(torch.zeros([1, vis_seq, dim],
                                                          dtype=torch.float32),
                                              requires_grad=True)
            trunc_normal_(self.vis_pos_embed, std=0.02)
        elif pos2d:
            pos_embed = torch.zeros([1, max_size[0] * max_size[1], dim],
                                    dtype=torch.float32)
            trunc_normal_(pos_embed, mean=0, std=0.02)
            self.vis_pos_embed = nn.Parameter(
                pos_embed.transpose(1, 2).reshape(1, dim, max_size[0],
                                                  max_size[1]),
                requires_grad=True,
            )
        self.prompt_pos_embed = nn.Parameter(torch.zeros([1, 6, dim],
                                                         dtype=torch.float32),
                                             requires_grad=True)

        self.answer_query = nn.Parameter(torch.zeros([1, 1, dim],
                                                     dtype=torch.float32),
                                         requires_grad=True)
        self.norm_pred = nn.LayerNorm(dim, eps=1e-6)
        self.ques1_head = nn.Linear(dim, self.out_channels - 2)
        self.ques2_head = nn.Linear(dim, self.max_len, bias=False)
        self.ques3_head = nn.Linear(dim, self.max_len - 1)
        self.ques4_head = nn.Linear(dim, self.max_len - 1)
        trunc_normal_(self.char_pos_embed, std=0.02)
        trunc_normal_(self.appear_num_embed, std=0.02)
        trunc_normal_(self.answer_query, std=0.02)
        trunc_normal_(self.prompt_pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {
            'char_pos_embed', 'vis_pos_embed', 'appear_num_embed',
            'answer_query', 'char_embed'
        }

    def question_encoder(self, targets, train_i):
        (
            prompt_pos_idx,
            prompt_char_idx,
            ques_pos_idx,
            ques1_answer,
            ques2_char_idx,
            ques2_answer,
            ques4_char_num,
            ques_len,
            ques2_len,
            prompt_len,
        ) = targets
        max_ques_len = torch.max(ques_len)
        max_ques2_len = torch.max(ques2_len)
        max_prompt_len = torch.max(prompt_len)
        if self.next_pred and (train_i == 2 or train_i == 3):
            prompt_pos = self.prompt_pos_embed
            prompt_char_idx = prompt_char_idx[:, :max_prompt_len]
        else:
            prompt_pos = F.embedding(
                prompt_pos_idx[:, :max_prompt_len], self.char_pos_embed
            )  # bs lp [ 0,  4,  3, 12, 12, 12, 12, 12, 12, 12, 12]
            prompt_char_idx = prompt_char_idx[:, :max_prompt_len]
        prompt_char = self.char_embed(prompt_char_idx)  # bs lp

        prompt = prompt_pos + prompt_char
        mask_1234 = torch.where(prompt_char_idx == self.ignore_index,
                                float('-inf'), 0)

        ques1 = F.embedding(ques_pos_idx[:, :max_ques_len],
                            self.char_pos_embed)  # bs lq1 dim
        ques1_answer = ques1_answer[:, :max_ques_len]
        if self.quesall or train_i == 0:
            ques2_char = self.char_embed(ques2_char_idx[:, :max_ques2_len, 1])
            ques2 = ques2_char + F.embedding(ques2_char_idx[:, :max_ques2_len,
                                                            0],
                                             self.char_pos_embed)  # bs lq2 dim
            ques2_answer = ques2_answer[:, :max_ques2_len]
            ques2_head = F.embedding(ques2_char_idx[:, :max_ques2_len, 0],
                                     self.ques2_head.weight)
            ques4_char = self.char_embed(ques1_answer)
            ques4_ap_num = F.embedding(ques4_char_num[:, :max_ques_len],
                                       self.appear_num_embed)
            ques4 = ques4_char + ques4_ap_num
            ques4_answer = ques_pos_idx[:, :max_ques_len]

            return (
                prompt,
                ques1,
                ques2,
                ques2_head,
                ques4,
                ques1_answer,
                ques2_answer,
                ques4_answer,
                mask_1234.unsqueeze(1),
            )
        else:
            return prompt, ques1, ques1_answer, mask_1234.unsqueeze(1)

    def forward(self, x, data=None):
        if self.training:
            return self.forward_train(x, data)
        else:
            return self.forward_test(x)

    def forward_test(self, x):
        """
        Perform the forward pass for the test phase.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor or List[torch.Tensor]: The output logits or a list containing predicted indices and probabilities.

        The function handles different modes of operation based on the attributes:
        - `self.ds`: Determines if positional embedding is added to the input tensor.
        - `self.pos2d`: Determines if the positional embedding is 2D.
        - `self.ar`: Determines if autoregressive decoding is used.
        - `self.check_search`: Determines if beam search is used.
        - `self.next_pred`: Determines if next token prediction is used.
        - `self.refine_iter`: Number of refinement iterations for the predictions.

        The function performs the following steps:
        1. Adds positional embeddings to the input tensor if required.
        2. Initializes the BOS (beginning of sequence) prompt.
        3. Depending on the mode, performs decoding using different strategies:
            - Beam search decoding.
            - Autoregressive decoding.
            - Next token prediction.
        4. If refinement iterations are specified, refines the predictions.
        5. Returns the final logits or the predicted indices and probabilities.
        """
        if not self.ds:
            visual_f = x + self.vis_pos_embed
        elif self.pos2d:
            x = x + self.vis_pos_embed[:, :, :x.shape[2], :x.shape[3]]
            visual_f = x.flatten(2).transpose(1, 2)
        else:
            visual_f = x
        bs = x.shape[0]
        prompt_bos = self.char_embed(
            torch.full(
                [bs, 1], self.bos, dtype=torch.long,
                device=x.get_device())) + self.char_pos_embed[:1, :].unsqueeze(
                    0)  # BOS prompt
        ques_all = torch.tile(self.char_pos_embed.unsqueeze(0), (bs, 1, 1))
        if not self.ar:
            if self.check_search:
                tgt_in = torch.full((bs, self.max_len),
                                    self.ignore_index,
                                    dtype=torch.long,
                                    device=x.get_device())
                tgt_in[:, 0] = self.bos
                logits = []
                for j in range(1, self.max_len):
                    visual_f_check = visual_f
                    ques_check_i = ques_all[:, j:j + 1, :] + self.char_embed(
                        torch.arange(self.out_channels - 2,
                                     device=x.get_device())).unsqueeze(0)
                    prompt_check = ques_all[:, :j] + self.char_embed(
                        tgt_in[:, :j])
                    # prompt_check = prompt_bos
                    mask = torch.where(
                        (tgt_in[:, :j] == self.eos).int().cumsum(-1) > 0,
                        float('-inf'), 0)
                    for layer in self.cmff_decoder:
                        ques_check_i, prompt_check, visual_f_check = layer(
                            ques_check_i, prompt_check, visual_f_check,
                            mask.unsqueeze(1))
                    answer_query_i = self.answer_to_question_layer(
                        ques_check_i, prompt_check, mask.unsqueeze(1))
                    answer_pred_i = self.norm_pred(
                        self.answer_to_image_layer(
                            answer_query_i, visual_f_check))  # B, 26, 37
                    # the next token probability is in the output's ith token position
                    fc_2 = self.ques2_head.weight[j:j + 1].unsqueeze(0)
                    fc_2 = fc_2.tile([bs, 1, 1])
                    p_i = fc_2 @ answer_pred_i.transpose(1, 2)
                    # p_i = p_i[:, 0, :]
                    logits.append(p_i)
                    if j < self.max_len - 1:
                        # greedy decode. add the next token index to the target input
                        tgt_in[:, j] = p_i.squeeze().argmax(-1)
                        # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                        if (tgt_in == self.eos).any(dim=-1).all():
                            break
                logits = torch.cat(logits, dim=1)
            else:
                ques_pd = ques_all[:, 1:, :]
                prompt_pd = prompt_bos
                visual_f_pd = visual_f
                for layer in self.cmff_decoder:
                    ques_pd, prompt_pd, visual_f_pd = layer(
                        ques_pd, prompt_pd, visual_f_pd)
                answer_query_pd = self.answer_to_question_layer(
                    ques_pd, prompt_pd)
                answer_feats_pd = self.norm_pred(
                    self.answer_to_image_layer(answer_query_pd,
                                               visual_f_pd))  # B, 26, 37
                logits = self.ques1_head(answer_feats_pd)
        elif self.next_pred:
            ques_pd_1 = ques_all[:, 1:2, :]
            prompt_pd = prompt_bos
            visual_f_pd = visual_f
            for layer in self.cmff_decoder:
                ques_pd_1, prompt_pd, visual_f_pd = layer(
                    ques_pd_1, prompt_pd, visual_f_pd)
            answer_query_pd = self.answer_to_question_layer(
                ques_pd_1, prompt_pd)
            answer_feats_pd = self.norm_pred(
                self.answer_to_image_layer(answer_query_pd,
                                           visual_f_pd))  # B, 26, 37
            logits_pd_1 = self.ques1_head(answer_feats_pd)

            ques_next = self.char_pos_embed[-2:-1, :].unsqueeze(0).tile(
                [bs, 1, 1])
            prompt_next_bos = (self.char_embed(
                torch.full(
                    [bs, 1], self.bos, dtype=torch.long,
                    device=x.get_device())) + self.prompt_pos_embed[:, :1, :])
            pred_prob, pred_id = F.softmax(logits_pd_1, -1).max(-1)
            pred_prob_list = [pred_prob]
            pred_id_list = [pred_id]
            for j in range(1, 70):
                prompt_next_1 = self.char_embed(
                    pred_id) + self.prompt_pos_embed[:,
                                                     -1 * pred_id.shape[1]:, :]
                prompt_next = torch.concat([prompt_next_bos, prompt_next_1], 1)
                ques_next_i = ques_next
                visual_f_i = visual_f
                for layer in self.cmff_decoder:
                    ques_next_i, prompt_next, visual_f_pd = layer(
                        ques_next_i, prompt_next, visual_f_i)
                answer_query_next_i = self.answer_to_question_layer(
                    ques_next_i, prompt_next)
                answer_feats_next_i = self.norm_pred(
                    self.answer_to_image_layer(answer_query_next_i,
                                               visual_f_i))  # B, 26, 37
                logits_next_i = self.ques1_head(answer_feats_next_i)
                # pred_id = logits_next_i.argmax(-1)
                pred_prob_i, pred_id_i = F.softmax(logits_next_i, -1).max(-1)
                pred_prob_list.append(pred_prob_i)
                pred_id_list.append(pred_id_i)
                if (torch.concat(pred_id_list,
                                 1) == self.eos).any(dim=-1).all():
                    break
                if pred_id.shape[1] >= 5:
                    pred_id = torch.concat([pred_id[:, 1:], pred_id_i], 1)
                else:
                    pred_id = torch.concat([pred_id, pred_id_i], 1)
            return [
                torch.concat(pred_id_list, 1),
                torch.concat(pred_prob_list, 1)
            ]

        else:
            tgt_in = torch.full((bs, self.max_len),
                                self.ignore_index,
                                dtype=torch.long,
                                device=x.get_device())
            tgt_in[:, 0] = self.bos
            logits = []
            for j in range(1, self.max_len):
                visual_f_ar = visual_f
                ques_i = ques_all[:, j:j + 1, :]
                prompt_ar = ques_all[:, :j] + self.char_embed(tgt_in[:, :j])
                mask = torch.where(
                    (tgt_in[:, :j] == self.eos).int().cumsum(-1) > 0,
                    float('-inf'), 0)
                for layer in self.cmff_decoder:
                    ques_i, prompt_ar, visual_f_ar = layer(
                        ques_i, prompt_ar, visual_f_ar, mask.unsqueeze(1))
                answer_query_i = self.answer_to_question_layer(
                    ques_i, prompt_ar, mask.unsqueeze(1))
                answer_pred_i = self.norm_pred(
                    self.answer_to_image_layer(answer_query_i,
                                               visual_f_ar))  # B, 26, 37
                # the next token probability is in the output's ith token position
                p_i = self.ques1_head(answer_pred_i)
                logits.append(p_i)
                if j < self.max_len - 1:
                    # greedy decode. add the next token index to the target input
                    tgt_in[:, j] = p_i.squeeze().argmax(-1)

                    # Efficient batch decoding: If all output words have at least one EOS token, end decoding.
                    if (tgt_in == self.eos).any(dim=-1).all():
                        break
            logits = torch.cat(logits, dim=1)

        if self.refine_iter > 0:
            pred_probs, pred_idxs = F.softmax(logits, -1).max(-1)
            for i in range(self.refine_iter):

                mask_check = (pred_idxs == self.eos).int().cumsum(-1) <= 1

                ques_check_all = self.char_embed(
                    pred_idxs) + ques_all[:, 1:pred_idxs.shape[1] + 1, :]
                prompt_check = prompt_bos
                visual_f_check = visual_f
                ques_check = ques_check_all
                for layer in self.cmff_decoder:
                    ques_check, prompt_check, visual_f_check = layer(
                        ques_check, prompt_check, visual_f_check)
                answer_query_check = self.answer_to_question_layer(
                    ques_check, prompt_check)
                answer_pred_check = self.norm_pred(
                    self.answer_to_image_layer(answer_query_check,
                                               visual_f_check))  # B, 26, 37
                ques2_head = self.ques2_head.weight[1:pred_idxs.shape[1] +
                                                    1, :]
                ques2_head = torch.tile(ques2_head.unsqueeze(0), [bs, 1, 1])
                answer2_pred = answer_pred_check.matmul(
                    ques2_head.transpose(1, 2))
                diag_mask = torch.eye(answer2_pred.shape[1],
                                      device=x.get_device()).unsqueeze(0).tile(
                                          [bs, 1, 1])
                answer2_pred = F.sigmoid(
                    (answer2_pred * diag_mask).sum(-1)) * mask_check

                check_result = answer2_pred < 0.9  # pred_probs < 0.99

                prompt_refine = torch.concat([prompt_bos, ques_check_all], 1)
                mask_refine = torch.where(
                    check_result, float('-inf'), 0) + torch.where(
                        (pred_idxs == self.eos).int().cumsum(-1) < 1, 0,
                        float('-inf'))
                mask_refine = torch.concat(
                    [torch.zeros([bs, 1], device=x.get_device()), mask_refine],
                    1).unsqueeze(1)
                ques_refine = ques_all[:, 1:pred_idxs.shape[1] + 1, :]
                visual_f_refine = visual_f
                for layer in self.cmff_decoder:
                    ques_refine, prompt_refine, visual_f_refine = layer(
                        ques_refine, prompt_refine, visual_f_refine,
                        mask_refine)
                answer_query_refine = self.answer_to_question_layer(
                    ques_refine, prompt_refine, mask_refine)
                answer_pred_refine = self.norm_pred(
                    self.answer_to_image_layer(answer_query_refine,
                                               visual_f_refine))  # B, 26, 37
                answer_refine = self.ques1_head(answer_pred_refine)
                refine_probs, refine_idxs = F.softmax(answer_refine,
                                                      -1).max(-1)
                pred_idxs_refine = torch.where(check_result, refine_idxs,
                                               pred_idxs)
                pred_idxs = torch.where(mask_check, pred_idxs_refine,
                                        pred_idxs)
                pred_probs_refine = torch.where(check_result, refine_probs,
                                                pred_probs)
                pred_probs = torch.where(mask_check, pred_probs_refine,
                                         pred_probs)

            return [pred_idxs, pred_probs]

        return F.softmax(logits, -1)

    def forward_train(self, x, targets=None):
        """
        Forward pass for training the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, ...).
            targets (list, optional): List of target tensors. The list should contain:
                - targets[1]: Tensor of shape (batch_size, ...), prompt position indices.
                - targets[2]: Tensor of shape (batch_size, ...), prompt character indices.
                - targets[3]: Tensor of shape (batch_size, ...), question position indices.
                - targets[4]: Tensor of shape (batch_size, ...), question 1 answers.
                - targets[5]: Tensor of shape (batch_size, ...), question 2 character indices.
                - targets[6]: Tensor of shape (batch_size, ...), question 2 answers.
                - targets[7]: Tensor of shape (batch_size, ..., 2), question 3 character indices and answers.
                - targets[8]: Tensor of shape (batch_size, ...), question 4 character numbers.
                - targets[9]: Tensor of shape (batch_size, ...), question lengths.
                - targets[10]: Tensor of shape (batch_size, ...), prompt lengths.
                - targets[11]: Tensor of shape (batch_size, ...), question 4 answers.

        Returns:
            list: A list containing:
                - loss (dict): Dictionary containing the total loss and individual losses for each question.
                    - 'loss': Total loss.
                    - 'loss1': Loss for question 1.
                    - 'loss2': Loss for question 2.
                    - 'loss3': Loss for question 3.
                    - 'loss4': Loss for question 4.
                - logits (torch.Tensor): Logits for question 1 predictions.
        """

        bs = x.shape[0]
        answer_token = torch.tile(self.answer_query, (bs, 1, 1))
        if self.ch:
            ques3 = self.char_embed(targets[7][:, :,
                                               0]) + answer_token  # bs nc dim
            ques3_answer = targets[7][:, :, 1]
        else:
            ques3 = self.char_embed(
                torch.arange(self.out_channels - 2, device=x.get_device())
            ).unsqueeze(0) + answer_token  # bs nc dim
            ques3_answer = targets[7]
        loss1_list = []
        loss2_list = []
        loss3_list = []
        loss4_list = []
        sampler1_num = 0
        sampler2_num = 0
        sampler3_num = 0
        sampler4_num = 0
        if not self.ds:
            visual_f = x + self.vis_pos_embed
        elif self.pos2d:
            x = x + self.vis_pos_embed[:, :, :x.shape[2], :x.shape[3]]
            visual_f = x.flatten(2).transpose(1, 2)
        else:
            visual_f = x
        train_i = 0
        for target_ in zip(
                targets[1].transpose(0, 1),
                targets[2].transpose(0, 1),
                targets[3].transpose(0, 1),
                targets[4].transpose(0, 1),
                targets[5].transpose(0, 1),
                targets[6].transpose(0, 1),
                targets[8].transpose(0, 1),
                targets[9].transpose(0, 1),
                targets[10].transpose(0, 1),
                targets[11].transpose(0, 1),
        ):
            # target_ = [prompt_pos_idx, prompt_char_idx, ques_pos_idx, ques1_answer, \
            # ques2_char_idx, ques2_answer, ques4_char_num, ques_len, prompt_len]
            visual_f_1234 = visual_f
            if self.quesall or train_i == 0:
                (
                    prompt,
                    ques1,
                    ques2,
                    ques2_head,
                    ques4,
                    ques1_answer,
                    ques2_answer,
                    ques4_answer,
                    mask_1234,
                ) = self.question_encoder(target_, train_i)
                prompt_1234 = prompt
                ques_1234 = torch.concat([ques1, ques2, ques3, ques4], 1)
                for layer in self.cmff_decoder:
                    ques_1234, prompt_1234, visual_f_1234 = layer(
                        ques_1234, prompt_1234, visual_f_1234, mask_1234)
                answer_query_1234 = self.answer_to_question_layer(
                    ques_1234, prompt_1234, mask_1234)
                answer_feats_1234 = self.norm_pred(
                    self.answer_to_image_layer(answer_query_1234,
                                               visual_f_1234))  # B, 26, 37

                answer_feats_1 = answer_feats_1234[:, :ques1.shape[1], :]
                answer_feats_2 = answer_feats_1234[:, ques1.shape[1]:(
                    ques1.shape[1] + ques2.shape[1]), :]
                answer_feats_3 = answer_feats_1234[:, (
                    ques1.shape[1] + ques2.shape[1]):-ques4.shape[1], :]
                answer_feats_4 = answer_feats_1234[:, -ques4.shape[1]:, :]

                answer1_pred = self.ques1_head(answer_feats_1)
                if train_i == 0:
                    logits = answer1_pred

                n = (ques1_answer != self.ignore_index).sum().item()
                loss1 = n * F.cross_entropy(
                    answer1_pred.flatten(0, 1),
                    ques1_answer.flatten(0, 1),
                    ignore_index=self.ignore_index,
                    reduction='mean',
                )
                sampler1_num += n
                loss1_list.append(loss1)

                answer2_pred = answer_feats_2.matmul(ques2_head.transpose(
                    1, 2))
                diag_mask = torch.eye(answer2_pred.shape[1],
                                      device=x.get_device()).unsqueeze(0).tile(
                                          [bs, 1, 1])
                answer2_pred = (answer2_pred * diag_mask).sum(-1)

                ques2_answer = ques2_answer.flatten(0, 1)
                non_pad_mask = torch.not_equal(ques2_answer, self.ignore_index)
                n = non_pad_mask.sum().item()
                ques2_answer = torch.where(ques2_answer == self.ignore_index,
                                           0, ques2_answer)
                loss2_none = F.binary_cross_entropy_with_logits(
                    answer2_pred.flatten(0, 1), ques2_answer, reduction='none')
                loss2 = n * loss2_none.masked_select(non_pad_mask).mean()
                sampler2_num += n
                loss2_list.append(loss2)

                answer3_pred = self.ques3_head(answer_feats_3)
                n = (ques3_answer != self.ignore_index).sum().item()
                loss3 = n * F.cross_entropy(answer3_pred.flatten(0, 1),
                                            ques3_answer.flatten(0, 1),
                                            reduction='mean')
                sampler3_num += n
                loss3_list.append(loss3)

                answer4_pred = self.ques4_head(answer_feats_4)
                n = (ques4_answer != self.max_len - 1).sum().item()
                loss4 = n * F.cross_entropy(
                    answer4_pred.flatten(0, 1),
                    ques4_answer.flatten(0, 1),
                    ignore_index=self.max_len - 1,
                    reduction='mean',
                )
                sampler4_num += n
                loss4_list.append(loss4)
            else:
                prompt, ques1, ques1_answer, mask_1234 = self.question_encoder(
                    target_, train_i)
                prompt_1234 = prompt
                for layer in self.cmff_decoder:
                    ques1, prompt_1234, visual_f_1234 = layer(
                        ques1, prompt_1234, visual_f_1234, mask_1234)
                answer_query_1 = self.answer_to_question_layer(
                    ques1, prompt_1234, mask_1234)
                answer_feats_1 = self.norm_pred(
                    self.answer_to_image_layer(answer_query_1,
                                               visual_f_1234))  # B, 26, 37
                answer1_pred = self.ques1_head(answer_feats_1)
                n = (ques1_answer != self.ignore_index).sum().item()
                loss1 = n * F.cross_entropy(
                    answer1_pred.flatten(0, 1),
                    ques1_answer.flatten(0, 1),
                    ignore_index=self.ignore_index,
                    reduction='mean',
                )
                sampler1_num += n
                loss1_list.append(loss1)
            train_i += 1

        loss_list = [
            sum(loss1_list) / sampler1_num,
            sum(loss2_list) / sampler2_num,
            sum(loss3_list) / sampler3_num,
            sum(loss4_list) / sampler4_num,
        ]
        loss = {
            'loss': sum(loss_list),
            'loss1': loss_list[0],
            'loss2': loss_list[1],
            'loss3': loss_list[2],
            'loss4': loss_list[3],
        }
        return [loss, logits]
