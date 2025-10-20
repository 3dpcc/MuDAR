import torch
import torch.nn as nn

from model.knn_transformer import KNNCrossAttentionBlock


class PosEmb(nn.Module):

    def __init__(self, k):
        super().__init__()

        self.encoder_0 = nn.Embedding(256, 28)
        self.encoder_1 = nn.Embedding(16, 2)
        self.encoder_2 = nn.Embedding(8, 2)

        self.encoder_3 = nn.Embedding(32, 16)
        self.encoder_4 = nn.Embedding(2250, 16)

        self.pos_mlp  = nn.Linear(3, 128, bias=False)

        self.fuse_mlp = nn.Linear(128 * 3, 256)

        self.pct = KNNCrossAttentionBlock(256, k)

    def forward(self, occupy, level, octant, laser, phi, pos):
        # 32 channels
        occupy_emb = self.encoder_0(occupy)
        level_emb  = self.encoder_1(level)
        octant_emb = self.encoder_2(octant)
        laser_emb  = self.encoder_3(laser)
        phi_emb    = self.encoder_4(phi)

        # normalize position
        pos_min = torch.min(pos, dim=0, keepdim=True)[0]
        pos_max = torch.max(pos, dim=0, keepdim=True)[0]
        pos_norm = (pos - pos_min) / (pos_max - pos_min + 1e-7)

        pos_norm = self.pos_mlp(pos_norm)

        emb_1 = torch.cat((occupy_emb, level_emb, octant_emb), -1)
        emb_1 = emb_1.reshape((emb_1.shape[0], emb_1.shape[1], -1))
        emb_2 = torch.cat((laser_emb, phi_emb), -1)
        emb_2 = emb_2.reshape((emb_2.shape[0], emb_2.shape[1], -1))

        emb = torch.cat((emb_1, emb_2, pos_norm), dim=-1)
        emb = self.fuse_mlp(emb)

        pos_norm = pos_norm.permute(1, 0, 2)
        emb  = emb.permute(1, 0, 2)

        pos_emb = self.pct(pos_norm, emb)

        return pos_emb
