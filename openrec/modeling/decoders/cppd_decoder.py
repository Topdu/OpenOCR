import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import ones_, trunc_normal_, zeros_

from openrec.modeling.common import DropPath, Identity, Mlp
from openrec.modeling.decoders.nrtr_decoder import Embeddings


class Attention(nn.Module):

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
        # if not self.training:
        #     self.attn_map = attn
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
        self.norm1 = eval(norm_layer)(dim, epsilon=epsilon)
        self.norm2 = eval(norm_layer)(dim, epsilon=epsilon)

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
        norm_layer=nn.LayerNorm,
        epsilon=1e-6,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim, eps=epsilon)
        self.mixer = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        self.norm2 = norm_layer(dim, eps=epsilon)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, q, kv, key_mask=None):
        x1 = self.norm1(q + self.drop_path(self.mixer(q, kv, key_mask)))
        x = self.norm2(x1 + self.drop_path(self.mlp(x1)))
        return x


class CPPDDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_layer=2,
                 drop_path_rate=0.1,
                 max_len=25,
                 vis_seq=50,
                 iters=1,
                 pos_len=False,
                 ch=False,
                 rec_layer=1,
                 num_heads=None,
                 ds=False,
                 **kwargs):
        super(CPPDDecoder, self).__init__()

        self.out_channels = out_channels  # none + 26 + 10
        dim = in_channels
        self.dim = dim
        self.iters = iters
        self.max_len = max_len + 1  # max_len + eos
        self.pos_len = pos_len
        self.ch = ch
        self.char_node_embed = Embeddings(d_model=dim,
                                          vocab=self.out_channels,
                                          scale_embedding=True)
        self.pos_node_embed = Embeddings(d_model=dim,
                                         vocab=self.max_len,
                                         scale_embedding=True)
        dpr = np.linspace(0, drop_path_rate, num_layer + rec_layer)

        self.char_node_decoder = nn.ModuleList([
            DecoderLayer(
                dim=dim,
                num_heads=dim // 32 if num_heads is None else num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_path=dpr[i],
            ) for i in range(num_layer)
        ])
        self.pos_node_decoder = nn.ModuleList([
            DecoderLayer(
                dim=dim,
                num_heads=dim // 32 if num_heads is None else num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                drop_path=dpr[i],
            ) for i in range(num_layer)
        ])

        self.edge_decoder = nn.ModuleList([
            DecoderLayer(
                dim=dim,
                num_heads=dim // 32 if num_heads is None else num_heads,
                mlp_ratio=4.0,
                qkv_bias=True,
                qk_scale=1.0 if (rec_layer + i) % 2 != 0 else None,
                drop_path=dpr[num_layer + i],
            ) for i in range(rec_layer)
        ])
        self.rec_layer_num = rec_layer
        self_mask = torch.tril(
            torch.ones([self.max_len, self.max_len], dtype=torch.float32))
        self_mask = torch.where(
            self_mask > 0,
            torch.zeros_like(self_mask, dtype=torch.float32),
            torch.full([self.max_len, self.max_len],
                       float('-inf'),
                       dtype=torch.float32),
        )
        self.self_mask = self_mask.unsqueeze(0)
        self.char_pos_embed = nn.Parameter(torch.zeros([1, self.max_len, dim],
                                                       dtype=torch.float32),
                                           requires_grad=True)
        self.ds = ds
        if not self.ds:
            self.vis_pos_embed = nn.Parameter(torch.zeros([1, vis_seq, dim],
                                                          dtype=torch.float32),
                                              requires_grad=True)
            trunc_normal_(self.vis_pos_embed, std=0.02)
        self.char_node_fc1 = nn.Linear(dim, max_len)

        self.pos_node_fc1 = nn.Linear(dim, self.max_len)

        self.edge_fc = nn.Linear(dim, self.out_channels)

        trunc_normal_(self.char_pos_embed, std=0.02)

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
            'char_pos_embed', 'vis_pos_embed', 'char_node_embed',
            'pos_node_embed'
        }

    def forward(self, x, data=None):
        if self.training:
            return self.forward_train(x, data)
        else:
            return self.forward_test(x)

    def forward_test(self, x):
        if not self.ds:
            visual_feats = x + self.vis_pos_embed
        else:
            visual_feats = x
        bs = visual_feats.shape[0]

        pos_node_embed = self.pos_node_embed(
            torch.arange(self.max_len).cuda(
                x.get_device())).unsqueeze(0) + self.char_pos_embed
        pos_node_embed = torch.tile(pos_node_embed, [bs, 1, 1])

        char_vis_node_query = visual_feats
        pos_vis_node_query = torch.concat([pos_node_embed, visual_feats], 1)

        for char_decoder_layer, pos_decoder_layer in zip(
                self.char_node_decoder, self.pos_node_decoder):
            char_vis_node_query = char_decoder_layer(char_vis_node_query,
                                                     char_vis_node_query)
            pos_vis_node_query = pos_decoder_layer(
                pos_vis_node_query, pos_vis_node_query[:, self.max_len:, :])

        pos_node_query = pos_vis_node_query[:, :self.max_len, :]

        char_vis_feats = char_vis_node_query
        # pos_vis_feats = pos_vis_node_query[:, self.max_len :, :]

        # pos_node_feats = self.edge_decoder(
        #     pos_node_query, char_vis_feats, pos_vis_feats
        # )  # B, 26, dim

        pos_node_feats = pos_node_query
        for layer_i in range(self.rec_layer_num):
            rec_layer = self.edge_decoder[layer_i]
            if (self.rec_layer_num + layer_i) % 2 == 0:
                pos_node_feats = rec_layer(pos_node_feats, pos_node_feats,
                                           self.self_mask)
            else:
                pos_node_feats = rec_layer(pos_node_feats, char_vis_feats)
        edge_feats = self.edge_fc(pos_node_feats)  # B, 26, 37

        edge_logits = F.softmax(
            edge_feats,
            -1)  # * F.sigmoid(pos_node_feats1.unsqueeze(-1))  # B, 26, 37

        return edge_logits

    def forward_train(self, x, targets=None):
        if not self.ds:
            visual_feats = x + self.vis_pos_embed
        else:
            visual_feats = x
        bs = visual_feats.shape[0]

        if self.ch:
            char_node_embed = self.char_node_embed(targets[-2])
        else:
            char_node_embed = self.char_node_embed(
                torch.arange(self.out_channels).cuda(
                    x.get_device())).unsqueeze(0)
            char_node_embed = torch.tile(char_node_embed, [bs, 1, 1])
        counting_char_num = char_node_embed.shape[1]
        pos_node_embed = self.pos_node_embed(
            torch.arange(self.max_len).cuda(
                x.get_device())).unsqueeze(0) + self.char_pos_embed
        pos_node_embed = torch.tile(pos_node_embed, [bs, 1, 1])

        node_feats = []

        char_vis_node_query = torch.concat([char_node_embed, visual_feats], 1)
        pos_vis_node_query = torch.concat([pos_node_embed, visual_feats], 1)

        for char_decoder_layer, pos_decoder_layer in zip(
                self.char_node_decoder, self.pos_node_decoder):
            char_vis_node_query = char_decoder_layer(
                char_vis_node_query,
                char_vis_node_query[:, counting_char_num:, :])
            pos_vis_node_query = pos_decoder_layer(
                pos_vis_node_query, pos_vis_node_query[:, self.max_len:, :])

        char_node_query = char_vis_node_query[:, :counting_char_num, :]
        pos_node_query = pos_vis_node_query[:, :self.max_len, :]

        char_vis_feats = char_vis_node_query[:, counting_char_num:, :]

        char_node_feats1 = self.char_node_fc1(char_node_query)
        pos_node_feats1 = self.pos_node_fc1(pos_node_query)
        if not self.pos_len:
            diag_mask = torch.eye(pos_node_feats1.shape[1]).unsqueeze(0).tile(
                [pos_node_feats1.shape[0], 1, 1])
            pos_node_feats1 = (
                pos_node_feats1 *
                diag_mask.cuda(pos_node_feats1.get_device())).sum(-1)

        node_feats.append(char_node_feats1)
        node_feats.append(pos_node_feats1)

        pos_node_feats = pos_node_query
        for layer_i in range(self.rec_layer_num):
            rec_layer = self.edge_decoder[layer_i]
            if (self.rec_layer_num + layer_i) % 2 == 0:
                pos_node_feats = rec_layer(pos_node_feats, pos_node_feats,
                                           self.self_mask)
            else:
                pos_node_feats = rec_layer(pos_node_feats, char_vis_feats)
        edge_feats = self.edge_fc(pos_node_feats)  # B, 26, 37

        return node_feats, edge_feats
