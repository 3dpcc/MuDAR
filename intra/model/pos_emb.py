import torch
import torch.nn as nn

from model.knn_transformer import KNNCrossAttentionBlock


class PosEmb(nn.Module):

    def __init__(self, k):
        super().__init__()

        self.encoder0_128 = nn.Embedding(256, 24)
        self.encoder1_128 = nn.Embedding(16, 2)
        self.encoder2_128 = nn.Embedding(8, 2)
        self.encoder3_128 = nn.Embedding(32, 2)
        self.encoder4_128 = nn.Embedding(2250, 2)

        self.pos_mlp  = nn.Linear(3, 128, bias=False)

        self.pct = KNNCrossAttentionBlock(256, k)

    def forward(self, occupy, level, octant, laser, phi, pos):
        # 32 channels
        occupy_emb_128 = self.encoder0_128(occupy)
        level_emb_128  = self.encoder1_128(level)
        octant_emb_128 = self.encoder2_128(octant)
        laser_emb_128  = self.encoder3_128(laser)
        phi_emb_128    = self.encoder4_128(phi)

        # normalize position
        pos_min = torch.min(pos, dim=0, keepdim=True)[0]
        pos_max = torch.max(pos, dim=0, keepdim=True)[0]
        pos_norm = (pos - pos_min) / (pos_max - pos_min + 1e-7)

        pos_norm_128 = self.pos_mlp(pos_norm)

        emb_128 = torch.cat((occupy_emb_128, level_emb_128, octant_emb_128, laser_emb_128, phi_emb_128), -1)
        emb_128 = emb_128.reshape((emb_128.shape[0], emb_128.shape[1], -1))

        emb_128 = torch.cat((emb_128, pos_norm_128), dim=-1)

        pos_norm = pos_norm.permute(1, 0, 2)
        emb_128  = emb_128.permute(1, 0, 2)

        pos_emb = self.pct(pos_norm, emb_128)

        return pos_emb
