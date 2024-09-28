import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init


class Embedding(nn.Module):

    def __init__(self, in_timestep, in_planes, mid_dim=4096, embed_dim=300):
        super(Embedding, self).__init__()
        self.in_timestep = in_timestep
        self.in_planes = in_planes
        self.embed_dim = embed_dim
        self.mid_dim = mid_dim
        self.eEmbed = nn.Linear(
            in_timestep * in_planes,
            self.embed_dim)  # Embed encoder output to a word-embedding like

    def forward(self, x):
        x = x.flatten(1)
        x = self.eEmbed(x)
        return x


class Attn_Rnn_Block(nn.Module):

    def __init__(self, featdim, hiddendim, embedding_dim, out_channels,
                 attndim):
        super(Attn_Rnn_Block, self).__init__()

        self.attndim = attndim
        self.embedding_dim = embedding_dim
        self.feat_embed = nn.Linear(featdim, attndim)
        self.hidden_embed = nn.Linear(hiddendim, attndim)
        self.attnfeat_embed = nn.Linear(attndim, 1)
        self.gru = nn.GRU(input_size=featdim + self.embedding_dim,
                          hidden_size=hiddendim,
                          batch_first=True)
        self.fc = nn.Linear(hiddendim, out_channels)
        self.init_weights()

    def init_weights(self):
        init.normal_(self.hidden_embed.weight, std=0.01)
        init.constant_(self.hidden_embed.bias, 0)
        init.normal_(self.attnfeat_embed.weight, std=0.01)
        init.constant_(self.attnfeat_embed.bias, 0)

    def _attn(self, feat, h_state):
        b, t, _ = feat.shape
        feat = self.feat_embed(feat)
        h_state = self.hidden_embed(h_state.squeeze(0)).unsqueeze(1)
        h_state = h_state.expand(b, t, self.attndim)
        sumTanh = torch.tanh(feat + h_state)
        attn_w = self.attnfeat_embed(sumTanh).squeeze(-1)
        attn_w = F.softmax(attn_w, dim=1).unsqueeze(1)
        # [B,1,25]
        return attn_w

    def forward(self, feat, h_state, label_input):

        attn_w = self._attn(feat, h_state)

        attn_feat = attn_w @ feat
        attn_feat = attn_feat.squeeze(1)

        output, h_state = self.gru(
            torch.cat([label_input, attn_feat], 1).unsqueeze(1), h_state)
        pred = self.fc(output)

        return pred, h_state


class ASTERDecoder(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 embedding_dim=256,
                 hiddendim=256,
                 attndim=256,
                 max_len=25,
                 seed=False,
                 time_step=32,
                 **kwargs):
        super(ASTERDecoder, self).__init__()
        self.num_classes = out_channels
        self.bos = out_channels - 2
        self.eos = 0
        self.padding_idx = out_channels - 1
        self.seed = seed
        if seed:
            self.embeder = Embedding(
                in_timestep=time_step,
                in_planes=in_channels,
            )
        self.word_embedding = nn.Embedding(self.num_classes,
                                           embedding_dim,
                                           padding_idx=self.padding_idx)

        self.attndim = attndim
        self.hiddendim = hiddendim
        self.max_seq_len = max_len + 1

        self.featdim = in_channels

        self.attn_rnn_block = Attn_Rnn_Block(
            featdim=self.featdim,
            hiddendim=hiddendim,
            embedding_dim=embedding_dim,
            out_channels=out_channels - 2,
            attndim=attndim,
        )
        self.embed_fc = nn.Linear(300, self.hiddendim)

    def get_initial_state(self, embed, tile_times=1):
        assert embed.shape[1] == 300
        state = self.embed_fc(embed)  # N * sDim
        if tile_times != 1:
            state = state.unsqueeze(1)
            trans_state = state.transpose(0, 1)
            state = trans_state.tile([tile_times, 1, 1])
            trans_state = state.transpose(0, 1)
            state = trans_state.reshape(-1, self.hiddendim)
        state = state.unsqueeze(0)  # 1 * N * sDim
        return state

    def forward(self, feat, data=None):
        # b,25,512
        b = feat.size(0)
        if self.seed:
            embedding_vectors = self.embeder(feat)
            h_state = self.get_initial_state(embedding_vectors)
        else:
            h_state = torch.zeros(1, b, self.hiddendim).to(feat.device)
        outputs = []
        if self.training:
            label = data[0]
            label_embedding = self.word_embedding(label)  # [B,25,256]
            tokens = label_embedding[:, 0, :]
            max_len = data[1].max() + 1
        else:
            tokens = torch.full([b, 1],
                                self.bos,
                                device=feat.device,
                                dtype=torch.long)
            tokens = self.word_embedding(tokens.squeeze(1))
            max_len = self.max_seq_len
        pred, h_state = self.attn_rnn_block(feat, h_state, tokens)
        outputs.append(pred)

        dec_seq = torch.full((feat.shape[0], max_len),
                             self.padding_idx,
                             dtype=torch.int64,
                             device=feat.get_device())
        dec_seq[:, :1] = torch.argmax(pred, dim=-1)
        for i in range(1, max_len):
            if not self.training:
                max_idx = torch.argmax(pred, dim=-1).squeeze(1)
                tokens = self.word_embedding(max_idx)
                dec_seq[:, i] = max_idx
                if (dec_seq == self.eos).any(dim=-1).all():
                    break
            else:
                tokens = label_embedding[:, i, :]
            pred, h_state = self.attn_rnn_block(feat, h_state, tokens)
            outputs.append(pred)
        preds = torch.cat(outputs, 1)
        if self.seed and self.training:
            return [embedding_vectors, preds]
        return preds if self.training else F.softmax(preds, -1)
