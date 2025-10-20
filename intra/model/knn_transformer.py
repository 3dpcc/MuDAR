import math

import torch
from pytorch3d.ops import knn_gather, knn_points

class Attention(torch.nn.Module):
    def __init__(self, embed_dim):
        super().__init__()

        self.Q_linear = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.K_linear = torch.nn.Linear(embed_dim, embed_dim, bias=False)
        self.V_linear = torch.nn.Linear(embed_dim, embed_dim, bias=False)

        self.d = math.sqrt(embed_dim)

    def forward(self, src, tgt):
        Q = self.Q_linear(src)
        K = self.K_linear(tgt)
        V = self.V_linear(tgt)

        K = K.permute(0, 1, 3, 2)

        attention_map = torch.einsum("bndk, bnd -> bnk", [K, Q])
        attention_map = torch.softmax(attention_map / self.d, dim=-1)

        attention_feature = torch.einsum("bnk, bnkd -> bnd", [attention_map, V])

        return attention_feature


class KNNCrossAttentionBlock(torch.nn.Module):
    def __init__(self, channels, k=16):
        super().__init__()

        self.k = k

        self.attention_0 = Attention(channels)
        self.attention_1 = Attention(channels)

        self.linear = torch.nn.Linear(channels, channels)

        self.pos_linear_0 = torch.nn.Linear(3, channels)
        self.pos_linear_1 = torch.nn.Linear(channels, channels)
        self.layer_norm_0 = torch.nn.LayerNorm(channels)
        self.layer_norm_1 = torch.nn.LayerNorm(channels)

    def forward(self, src, tgt):
        if self.k > src.shape[1]:
            k = src.shape[1]
        else:
            k = self.k

        # idx = knn_points(src, src, K=self.k)[1]
        idx = torch.cdist(src, src).topk(k, dim=-1, largest=False)[1]

        gather_src = knn_gather(src, idx)
        residual_src = src.unsqueeze(2) - gather_src
        residual_src = self.pos_linear_0(residual_src)

        gather_tgt = knn_gather(tgt, idx)

        out = self.attention_0(tgt, residual_src + gather_tgt)

        # idx = knn_points(out, out, K=self.k)[1]
        idx = torch.cdist(out, out).topk(k, dim=-1, largest=False)[1]

        gather_out = knn_gather(out, idx)
        residual_out = out.unsqueeze(2) - gather_out
        residual_out = self.pos_linear_1(residual_out)

        out = self.attention_1(out, residual_out + gather_out)

        # skip-connettion
        out_0 = tgt + out
        out_1 = self.linear(self.layer_norm_0(out_0))
        out_2 = tgt + out_1
        out_2 = self.layer_norm_1(out_2)

        out_2 = out_2.permute(1, 0, 2)

        return out_2
