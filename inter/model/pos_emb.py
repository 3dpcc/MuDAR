import torch
import torch.nn as nn

from model.knn_transformer import KNNCrossAttentionBlock


class PositionEmbedding(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.prev_encoder_0 = nn.Embedding(cfg.ntoken, 24)
        self.prev_encoder_1 = nn.Embedding(cfg.nlevel, 2)
        self.prev_encoder_2 = nn.Embedding(cfg.noctant, 2)
        self.prev_encoder_3 = nn.Embedding(cfg.nlaser, 2)
        self.prev_encoder_4 = nn.Embedding(cfg.nphi, 2)

        self.curr_encoder_0 = nn.Embedding(cfg.ntoken, 24)
        self.curr_encoder_1 = nn.Embedding(cfg.nlevel, 2)
        self.curr_encoder_2 = nn.Embedding(cfg.noctant, 2)
        self.curr_encoder_3 = nn.Embedding(cfg.nlaser, 2)
        self.curr_encoder_4 = nn.Embedding(cfg.nphi, 2)

        self.prev_pos_mlp  = nn.Linear(3, 128, bias=False)
        self.curr_pos_mlp  = nn.Linear(3, 128, bias=False)

        self.pct = KNNCrossAttentionBlock(256, cfg.k)

    def forward(self, prev, curr):
        prev_occupy = prev[:, :,  :, 0]
        prev_level  = prev[:, :,  :, 1]
        prev_octant = prev[:, :,  :, 2]
        prev_laser  = prev[:, :,  :, 6]
        prev_phi    = prev[:, :,  :, 7]
        prev_pos    = prev[:, :, -1, 3:6].float()

        curr_occupy = curr[:, :,  :, 0]
        curr_level  = curr[:, :,  :, 1]
        curr_octant = curr[:, :,  :, 2]
        curr_laser  = curr[:, :,  :, 6]
        curr_phi    = curr[:, :,  :, 7]
        curr_pos    = curr[:, :, -1, 3:6].float()

        prev_occupy_emb = self.prev_encoder_0(prev_occupy)
        prev_level_emb  = self.prev_encoder_1(prev_level)
        prev_octant_emb = self.prev_encoder_2(prev_octant)
        prev_laser_emb  = self.prev_encoder_3(prev_laser)
        prev_phi_emb    = self.prev_encoder_4(prev_phi)

        curr_occupy_emb = self.curr_encoder_0(curr_occupy)
        curr_level_emb  = self.curr_encoder_1(curr_level)
        curr_octant_emb = self.curr_encoder_2(curr_octant)
        curr_laser_emb  = self.curr_encoder_3(curr_laser)
        curr_phi_emb    = self.curr_encoder_4(curr_phi)

        # normalize position
        prev_pos_min = torch.min(prev_pos, dim=0, keepdim=True)[0]
        prev_pos_max = torch.max(prev_pos, dim=0, keepdim=True)[0]
        prev_pos_norm = (prev_pos - prev_pos_min) / (prev_pos_max - prev_pos_min + 1e-7)

        curr_pos_min = torch.min(curr_pos, dim=0, keepdim=True)[0]
        curr_pos_max = torch.max(curr_pos, dim=0, keepdim=True)[0]
        curr_pos_norm = (curr_pos - curr_pos_min) / (curr_pos_max - curr_pos_min + 1e-7)

        prev_pos_norm_128 = self.prev_pos_mlp(prev_pos_norm)
        curr_pos_norm_128 = self.curr_pos_mlp(curr_pos_norm)

        prev_emb_128 = torch.cat((prev_occupy_emb, prev_level_emb, prev_octant_emb, prev_laser_emb, prev_phi_emb), -1)
        prev_emb_128 = prev_emb_128.reshape((prev_emb_128.shape[0], prev_emb_128.shape[1], -1))
        prev_emb_128 = torch.cat((prev_emb_128, prev_pos_norm_128), dim=-1)

        curr_emb_128 = torch.cat((curr_occupy_emb, curr_level_emb, curr_octant_emb, curr_laser_emb, curr_phi_emb), -1)
        curr_emb_128 = curr_emb_128.reshape((curr_emb_128.shape[0], curr_emb_128.shape[1], -1))
        curr_emb_128 = torch.cat((curr_emb_128, curr_pos_norm_128), dim=-1)

        prev_pos_norm = prev_pos_norm.permute(1, 0, 2)
        curr_pos_norm = curr_pos_norm.permute(1, 0, 2)

        prev_emb_128 = prev_emb_128.permute(1, 0, 2)
        curr_emb_128 = curr_emb_128.permute(1, 0, 2)

        pos_emb = self.pct(prev_emb_128, curr_emb_128, prev_pos_norm, curr_pos_norm)

        return pos_emb
