import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init


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
                 max_seq_len=25,
                 **kwargs):
        super(ASTERDecoder, self).__init__()
        self.num_classes = out_channels
        self.bos_eos_idx = out_channels - 2
        self.padding_idx = out_channels - 1

        self.word_embedding = nn.Embedding(self.num_classes,
                                           embedding_dim,
                                           padding_idx=self.padding_idx)

        self.attndim = attndim
        self.hiddendim = hiddendim
        self.max_seq_len = max_seq_len

        self.featdim = in_channels

        self.attn_rnn_block = Attn_Rnn_Block(
            featdim=self.featdim,
            hiddendim=hiddendim,
            embedding_dim=embedding_dim,
            out_channels=out_channels,
            attndim=attndim,
        )

    def forward(self, feat, data=None):
        # b,25,512
        b = feat.size(0)

        h_state = torch.zeros(1, b, self.hiddendim).to(feat.device)
        outputs = []
        if self.training:
            label = data[0]
            label_embedding = self.word_embedding(label)  # [B,25,256]
            tokens = label_embedding[:, 0, :]
        else:
            tokens = torch.full([b, 1],
                                self.bos_eos_idx,
                                device=feat.device,
                                dtype=torch.long)
            tokens = self.word_embedding(tokens.squeeze(1))
        pred, h_state = self.attn_rnn_block(feat, h_state, tokens)
        outputs.append(pred)

        for i in range(1, self.max_seq_len):
            if not self.training:
                pred = F.softmax(pred, -1)
                max_idx = torch.argmax(pred, dim=-1)
                tokens = self.word_embedding(max_idx.squeeze(1))
            else:
                tokens = label_embedding[:, i, :]
            pred, h_state = self.attn_rnn_block(feat, h_state, tokens)
            outputs.append(pred)
        preds = torch.cat(outputs, 1)
        return preds
