import torch
from torch import nn


class AttentionLayer(nn.Module):
    def __init__(self, ninp, nhead, nhid):
        super().__init__()

        self.attention = torch.nn.MultiheadAttention(ninp, nhead)

        self.ffn = nn.Sequential(
            nn.Linear(ninp, nhid),
            nn.GELU(),
            nn.Linear(nhid, ninp),
        )

        self.ln_0 = nn.LayerNorm(ninp)
        self.ln_1 = nn.LayerNorm(ninp)

    def forward(self, src, tgt):
        out = self.attention(src, tgt, tgt)[0]
        src = self.ln_0(src + out)

        out = self.ffn(src)
        src = self.ln_1(src + out)

        return src

class TemporalFusion(nn.Module):

    def __init__(self, ninp):
        super().__init__()

        self.attn = AttentionLayer(ninp, 4, ninp)

    def forward(self, prev, curr):
        return self.attn(curr, prev)
