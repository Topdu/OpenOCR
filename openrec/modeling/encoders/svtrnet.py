import numpy as np
import torch
from torch import nn
from torch.nn.init import kaiming_normal_, ones_, trunc_normal_, zeros_

from openrec.modeling.common import DropPath, Identity, Mlp


class ConvBNLayer(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        bias=False,
        groups=1,
        act=nn.GELU,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


class ConvMixer(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        HW=[8, 25],
        local_k=[3, 3],
    ):
        super().__init__()
        self.HW = HW
        self.dim = dim
        self.local_mixer = nn.Conv2d(dim,
                                     dim,
                                     local_k,
                                     1, [local_k[0] // 2, local_k[1] // 2],
                                     groups=num_heads)

    def forward(self, x):
        h = self.HW[0]
        w = self.HW[1]
        x = x.transpose(1, 2).reshape([x.shape[0], self.dim, h, w])
        x = self.local_mixer(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Attention(nn.Module):

    def __init__(
        self,
        dim,
        num_heads=8,
        mixer='Global',
        HW=None,
        local_k=[7, 11],
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.HW = HW
        if HW is not None:
            H = HW[0]
            W = HW[1]
            self.N = H * W
            self.C = dim
        if mixer == 'Local' and HW is not None:
            hk = local_k[0]
            wk = local_k[1]
            mask = torch.ones(H * W,
                              H + hk - 1,
                              W + wk - 1,
                              dtype=torch.float32,
                              requires_grad=False)
            for h in range(0, H):
                for w in range(0, W):
                    mask[h * W + w, h:h + hk, w:w + wk] = 0.0
            mask = mask[:, hk // 2:H + hk // 2, wk // 2:W + wk // 2].flatten(1)
            mask[mask >= 1] = -np.inf
            self.register_buffer('mask', mask[None, None, :, :])
        self.mixer = mixer

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # x = F.scaled_dot_product_attention(
        #         q, k, v,
        #         attn_mask=mask,
        #         dropout_p=self.attn_drop.p
        #     )
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        if self.mixer == 'Local':
            attn += self.mask
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        mixer='Global',
        local_mixer=[7, 11],
        HW=None,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer='nn.LayerNorm',
        eps=1e-6,
        prenorm=True,
    ):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, eps=eps)
        else:
            self.norm1 = norm_layer(dim)
        if mixer == 'Global' or mixer == 'Local':
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                mixer=mixer,
                HW=HW,
                local_k=local_mixer,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif mixer == 'Conv':
            self.mixer = ConvMixer(dim,
                                   num_heads=num_heads,
                                   HW=HW,
                                   local_k=local_mixer)
        else:
            raise TypeError('The mixer must be one of [Global, Local, Conv]')

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, eps=eps)
        else:
            self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.prenorm = prenorm

    def forward(self, x):
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        img_size=[32, 100],
        in_channels=3,
        embed_dim=768,
        sub_num=2,
        patch_size=[4, 4],
        mode='pope',
    ):
        super().__init__()
        num_patches = (img_size[1] // (2**sub_num)) * (img_size[0] //
                                                       (2**sub_num))
        self.img_size = img_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.norm = None
        if mode == 'pope':
            if sub_num == 2:
                self.proj = nn.Sequential(
                    ConvBNLayer(
                        in_channels=in_channels,
                        out_channels=embed_dim // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias=None,
                    ),
                    ConvBNLayer(
                        in_channels=embed_dim // 2,
                        out_channels=embed_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias=None,
                    ),
                )
            if sub_num == 3:
                self.proj = nn.Sequential(
                    ConvBNLayer(
                        in_channels=in_channels,
                        out_channels=embed_dim // 4,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias=None,
                    ),
                    ConvBNLayer(
                        in_channels=embed_dim // 4,
                        out_channels=embed_dim // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias=None,
                    ),
                    ConvBNLayer(
                        in_channels=embed_dim // 2,
                        out_channels=embed_dim,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        act=nn.GELU,
                        bias=None,
                    ),
                )
        elif mode == 'linear':
            self.proj = nn.Conv2d(1,
                                  embed_dim,
                                  kernel_size=patch_size,
                                  stride=patch_size)
            self.num_patches = img_size[0] // patch_size[0] * img_size[
                1] // patch_size[1]

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class SubSample(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        types='Pool',
        stride=[2, 1],
        sub_norm='nn.LayerNorm',
        act=None,
    ):
        super().__init__()
        self.types = types
        if types == 'Pool':
            self.avgpool = nn.AvgPool2d(kernel_size=[3, 5],
                                        stride=stride,
                                        padding=[1, 2])
            self.maxpool = nn.MaxPool2d(kernel_size=[3, 5],
                                        stride=stride,
                                        padding=[1, 2])
            self.proj = nn.Linear(in_channels, out_channels)
        else:
            self.conv = nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size=3,
                                  stride=stride,
                                  padding=1)
        self.norm = eval(sub_norm)(out_channels)
        if act is not None:
            self.act = act()
        else:
            self.act = None

    def forward(self, x):
        if self.types == 'Pool':
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = (x1 + x2) * 0.5
            out = self.proj(x.flatten(2).transpose(1, 2))
        else:
            x = self.conv(x)
            out = x.flatten(2).transpose(1, 2)
        out = self.norm(out)
        if self.act is not None:
            out = self.act(out)

        return out


class SVTRNet(nn.Module):

    def __init__(
        self,
        img_size=[32, 100],
        in_channels=3,
        embed_dim=[64, 128, 256],
        depth=[3, 6, 3],
        num_heads=[2, 4, 8],
        mixer=['Local'] * 6 +
        ['Global'] * 6,  # Local atten, Global atten, Conv
        local_mixer=[[7, 11], [7, 11], [7, 11]],
        patch_merging='Conv',  # Conv, Pool, None
        sub_k=[[2, 1], [2, 1]],
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        last_drop=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer='nn.LayerNorm',
        sub_norm='nn.LayerNorm',
        eps=1e-6,
        out_channels=192,
        out_char_num=25,
        block_unit='Block',
        act='nn.GELU',
        last_stage=True,
        sub_num=2,
        prenorm=True,
        use_lenhead=False,
        feature2d=False,
        **kwargs,
    ):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.prenorm = prenorm
        self.feature2d = feature2d
        patch_merging = None if patch_merging != 'Conv' and patch_merging != 'Pool' else patch_merging
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            in_channels=in_channels,
            embed_dim=embed_dim[0],
            sub_num=sub_num,
        )
        num_patches = self.patch_embed.num_patches
        self.HW = [img_size[0] // (2**sub_num), img_size[1] // (2**sub_num)]
        self.hw = [
            [self.HW[0] // sub_k[0][0], self.HW[1] // sub_k[0][1]],
            [
                self.HW[0] // (sub_k[0][0] * sub_k[1][0]),
                self.HW[1] // (sub_k[0][1] * sub_k[1][1])
            ],
        ]
        # self.pos_embed = self.create_parameter(
        #     shape=[1, num_patches, embed_dim[0]], default_initializer=zeros_)
        # self.add_parameter("pos_embed", self.pos_embed)
        self.pos_embed = nn.Parameter(
            torch.zeros([1, num_patches, embed_dim[0]], dtype=torch.float32),
            requires_grad=True,
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        Block_unit = eval(block_unit)

        dpr = np.linspace(0, drop_path_rate, sum(depth))
        self.blocks1 = nn.ModuleList([
            Block_unit(
                dim=embed_dim[0],
                num_heads=num_heads[0],
                mixer=mixer[0:depth[0]][i],
                HW=self.HW,
                local_mixer=local_mixer[0],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=eval(act),
                attn_drop=attn_drop_rate,
                drop_path=dpr[0:depth[0]][i],
                norm_layer=norm_layer,
                eps=eps,
                prenorm=prenorm,
            ) for i in range(depth[0])
        ])
        if patch_merging is not None:
            self.sub_sample1 = SubSample(
                embed_dim[0],
                embed_dim[1],
                sub_norm=sub_norm,
                stride=sub_k[0],
                types=patch_merging,
            )
            HW = self.hw[0]
        else:
            HW = self.HW
        self.patch_merging = patch_merging
        self.blocks2 = nn.ModuleList([
            Block_unit(
                dim=embed_dim[1],
                num_heads=num_heads[1],
                mixer=mixer[depth[0]:depth[0] + depth[1]][i],
                HW=HW,
                local_mixer=local_mixer[1],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=eval(act),
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth[0]:depth[0] + depth[1]][i],
                norm_layer=norm_layer,
                eps=eps,
                prenorm=prenorm,
            ) for i in range(depth[1])
        ])
        if patch_merging is not None:
            self.sub_sample2 = SubSample(
                embed_dim[1],
                embed_dim[2],
                sub_norm=sub_norm,
                stride=sub_k[1],
                types=patch_merging,
            )
            HW = self.hw[1]
        self.blocks3 = nn.ModuleList([
            Block_unit(
                dim=embed_dim[2],
                num_heads=num_heads[2],
                mixer=mixer[depth[0] + depth[1]:][i],
                HW=HW,
                local_mixer=local_mixer[2],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=eval(act),
                attn_drop=attn_drop_rate,
                drop_path=dpr[depth[0] + depth[1]:][i],
                norm_layer=norm_layer,
                eps=eps,
                prenorm=prenorm,
            ) for i in range(depth[2])
        ])
        self.last_stage = last_stage
        if last_stage:
            self.avg_pool = nn.AdaptiveAvgPool2d([1, out_char_num])
            self.last_conv = nn.Conv2d(
                in_channels=embed_dim[2],
                out_channels=self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            )
            self.hardswish = nn.Hardswish()
            self.dropout = nn.Dropout(p=last_drop)
        else:
            self.out_channels = embed_dim[2]
        if not prenorm:
            self.norm = eval(norm_layer)(embed_dim[-1], eps=eps)
        self.use_lenhead = use_lenhead
        if use_lenhead:
            self.len_conv = nn.Linear(embed_dim[2], self.out_channels)
            self.hardswish_len = nn.Hardswish()
            self.dropout_len = nn.Dropout(p=last_drop)

        trunc_normal_(self.pos_embed, mean=0, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        if isinstance(m, nn.Conv2d):
            kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'sub_sample1', 'sub_sample2', 'sub_sample3'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample1(
                x.transpose(1, 2).reshape(-1, self.embed_dim[0], self.HW[0],
                                          self.HW[1]))
        for blk in self.blocks2:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample2(
                x.transpose(1, 2).reshape(-1, self.embed_dim[1], self.hw[0][0],
                                          self.hw[0][1]))
        for blk in self.blocks3:
            x = blk(x)
        if not self.prenorm:
            x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.feature2d:
            x = x.transpose(1, 2).reshape(-1, self.embed_dim[2], self.hw[1][0],
                                          self.hw[1][1])
        if self.use_lenhead:
            len_x = self.len_conv(x.mean(1))
            len_x = self.dropout_len(self.hardswish_len(len_x))
        if self.last_stage:
            x = self.avg_pool(
                x.transpose(1, 2).reshape(-1, self.embed_dim[2], self.hw[1][0],
                                          self.hw[1][1]))
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)
            x = x.flatten(2).transpose(1, 2)
        if self.use_lenhead:
            return x, len_x
        return x
