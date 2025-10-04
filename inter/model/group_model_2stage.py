import math

import lightning as l
import torch
import torch.nn as nn
import yaml

from model.attention_model import AttentionModule
from model.pos_emb import PositionEmbedding
from model.temporal_fusion import TemporalFusion
from utils.utils import AverageMeter, accuracy

class AncestralModel(l.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.nout = cfg.nout

        self.total_loss = AverageMeter()
        self.total_acc1 = AverageMeter()
        self.total_acc5 = AverageMeter()

        self.spatial_attention = AttentionModule(cfg.ninp, cfg.nhead, cfg.nhid)

        self.prev_encoder_0 = nn.Embedding(cfg.ntoken, 48)
        self.prev_encoder_1 = nn.Embedding(cfg.nlevel, 4)
        self.prev_encoder_2 = nn.Embedding(cfg.noctant, 4)
        self.prev_encoder_3 = nn.Embedding(cfg.nlaser, 4)
        self.prev_encoder_4 = nn.Embedding(cfg.nphi, 4)

        self.curr_encoder_0 = nn.Embedding(cfg.ntoken, 48)
        self.curr_encoder_1 = nn.Embedding(cfg.nlevel, 4)
        self.curr_encoder_2 = nn.Embedding(cfg.noctant, 4)
        self.curr_encoder_3 = nn.Embedding(cfg.nlaser, 4)
        self.curr_encoder_4 = nn.Embedding(cfg.nphi, 4)

        self.pos_emb = PositionEmbedding(cfg)
        self.temporal_fusion = TemporalFusion(256)

        self.decoder = nn.Sequential(
            nn.Linear(cfg.ninp, cfg.nhid),
            nn.GELU(),
            nn.Linear(cfg.nhid, cfg.nout),
        )

        self.save_hyperparameters(logger=False)

    def forward(self, prev, curr):
        l, b, _, _ = curr.size()

        prev_occupy = prev[:, :,  :, 0]
        prev_level  = prev[:, :,  :, 1]
        prev_octant = prev[:, :,  :, 2]
        prev_laser  = prev[:, :,  :, 6]
        prev_phi    = prev[:, :,  :, 7]

        curr_occupy = curr[:, :,  :, 0]
        curr_level  = curr[:, :,  :, 1]
        curr_octant = curr[:, :,  :, 2]
        curr_laser  = curr[:, :,  :, 6]
        curr_phi    = curr[:, :,  :, 7]

        # geometric feature extraction
        pos_emb = self.pos_emb(prev, curr)

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

        prev_emb = torch.cat([prev_occupy_emb, prev_level_emb, prev_octant_emb, prev_laser_emb, prev_phi_emb], dim=-1).reshape(l, b, -1)
        curr_emb = torch.cat([curr_occupy_emb, curr_level_emb, curr_octant_emb, curr_laser_emb, curr_phi_emb], dim=-1).reshape(l, b, -1)

        emb = self.temporal_fusion(prev_emb, curr_emb)
        emb = torch.cat([emb, pos_emb], dim=-1)

        output = self.spatial_attention(emb, mask=None)  # (l, b, c)

        output = self.decoder(output)

        return output

    def process_batch_data(self, batch):
        prev, curr = batch[0], batch[1]

        prev = prev.permute(1, 0, 2, 3).clone()
        curr = curr.permute(1, 0, 2, 3).clone()

        target = curr[:, :, -1, 0].clone()
        target_0 = target[0::2]
        target_1 = target[1::2]

        curr_0 = curr.clone()
        curr_1 = curr.clone()

        curr_0[   :, :, -1, 0] = 255
        curr_1[1::2, :, -1, 0] = 255

        return prev, curr_0, curr_1, target_0, target_1, target

    def training_step(self, batch, batch_idx):
        prev, curr_0, curr_1, target_0, target_1, target = self.process_batch_data(batch)

        output_0 = self.forward(prev, curr_0)
        output_1 = self.forward(prev, curr_1)
        output_0 = output_0[0::2]
        output_1 = output_1[1::2]

        output = torch.zeros([self.cfg.ctx_win, self.cfg.batch_size, self.nout], device=prev.device)
        output[0::2] = output_0
        output[1::2] = output_1

        output   = output.reshape(-1, self.nout)
        output_0 = output_0.reshape(-1, self.nout)
        output_1 = output_1.reshape(-1, self.nout)

        target   = target.reshape(-1).long()
        target_0 = target_0.reshape(-1).long()
        target_1 = target_1.reshape(-1).long()

        criterion = nn.CrossEntropyLoss(label_smoothing=0.)

        loss_0 = criterion(output_0, target_0) / math.log(2)
        loss_1 = criterion(output_1, target_1) / math.log(2)
        loss = loss_0 + loss_1

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        self.total_loss.update(loss.item())
        self.total_acc1.update(acc1.item())
        self.total_acc5.update(acc5.item())

        if batch_idx % self.cfg.log_on_bar_interval == 0:
            self.log("loss", self.total_loss.avg, prog_bar=True)
            self.log("acc1", self.total_acc1.avg, prog_bar=True)
            self.log("acc5", self.total_acc5.avg, prog_bar=True)

            self.log("lr",   self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)

            self.total_loss.reset()
            self.total_acc1.reset()
            self.total_acc5.reset()

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.99)

        return [optimizer], [scheduler]

    # def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
    #     if self.trainer.global_step < 2500:
    #         lr_scale = min(1.0, float(self.trainer.global_step + 1) / 5000.0 + 0.5)
    #         for pg in optimizer.param_groups:
    #             pg["lr"] = lr_scale * self.cfg.lr
    #     else:
    #         for pg in optimizer.param_groups:
    #             pg["lr"] = min(self.cfg.lr / 10, pg["lr"])

    #     optimizer.step(closure=optimizer_closure)


    def load_pretrained(self, path):
        state_dict_pre = torch.load(path, map_location="cuda")
        # state_dict_pre = torch.load(path, map_location="cuda")["state_dict"]
        state_dict_ref = self.state_dict()

        # keys = list(state_dict_pre.keys())

        # for key in keys:
        #     if '_orig_mod.' in key:
        #         del_key = key.replace('_orig_mod.', '')
        #         state_dict_pre[del_key] = state_dict_pre[key]
        #         del state_dict_pre[key]

        # keys = list(state_dict_pre.keys())

        # for key in keys:
        #     if 'global_attention.share_layers' in key:
        #         del_key = key.replace('global_attention.share_layers', 'spatial_attention.layers_1')
        #         state_dict_pre[del_key] = state_dict_pre[key]
        #         del state_dict_pre[key]

        #     if 'global_attention.layers' in key:
        #         del_key = key.replace('global_attention.layers', 'spatial_attention.layers_0')
        #         state_dict_pre[del_key] = state_dict_pre[key]
        #         del state_dict_pre[key]

        #     if 'decoder_0' in key:
        #         del_key = key.replace('decoder_0', 'decoder')
        #         state_dict_pre[del_key] = state_dict_pre[key]
        #         del state_dict_pre[key]

        # keys = list(state_dict_pre.keys())

        # for key in keys:
        #     if 'sequnetial0' in key:
        #         del_key = key.replace('sequnetial0', 'ffn')
        #         state_dict_pre[del_key] = state_dict_pre[key]
        #         del state_dict_pre[key]

        # keys = list(state_dict_pre.keys())

        # for key in keys:
        #     if 'pos_emb' in key:
        #         del state_dict_pre[key]

        # state_dict_pre["prev_encoder_3.weight"] = state_dict_pre["prev_encoder_3.weight"][:32]
        # state_dict_pre["curr_encoder_3.weight"] = state_dict_pre["curr_encoder_3.weight"][:32]
        # state_dict_pre["pos_emb.prev_encoder_3.weight"] = state_dict_pre["pos_emb.prev_encoder_3.weight"][:32]
        # state_dict_pre["pos_emb.curr_encoder_3.weight"] = state_dict_pre["pos_emb.curr_encoder_3.weight"][:32]

        keys = list(state_dict_pre.keys())

        for k in keys:
            if k not in state_dict_ref or state_dict_pre[k].shape != state_dict_ref[k].shape:
                state_dict_pre.pop(k)

        # torch.save(state_dict_pre, './ckpt/dynamic_kitti.ckpt')
        return self.load_state_dict(state_dict_pre, strict=True)
