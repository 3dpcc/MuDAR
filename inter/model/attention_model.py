import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    def __init__(self, ninp, nhead, nhid):
        super().__init__()

        self.attention = torch.nn.MultiheadAttention(ninp, nhead)

        self.ffn = nn.Sequential(
            nn.Linear(ninp, nhid),
            nn.GELU(),
            nn.Linear(nhid, ninp),
        )

        self.ln0 = nn.LayerNorm(ninp)
        self.ln1 = nn.LayerNorm(ninp)

    def forward(self, src, mask):
        out = self.attention(src, src, src, attn_mask=mask)[0]
        src = self.ln0(src + out)

        out = self.ffn(src)
        src = self.ln1(src + out)

        return src


class AttentionModule(nn.Module):
    def __init__(self, ninp, nhead, nhid):
        super().__init__()

        self.layers_0 = nn.ModuleList([
            AttentionLayer(ninp, nhead, nhid)
            for _ in range(4)
        ])

        self.layers_1 = nn.ModuleList([
            AttentionLayer(ninp, nhead, nhid)
            for _ in range(4)
        ])

    def forward(self, src, mask):
        for layer in self.layers_0:
            src = layer(src, mask)

        for layer in self.layers_1:
            for _ in range(3):
                src = layer(src, mask)

        return src
