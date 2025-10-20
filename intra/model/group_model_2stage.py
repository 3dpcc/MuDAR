import math

import lightning as l
import torch
import torch.nn as nn

from model.attention_model import AttentionModule
from model.pos_emb import PosEmb
from utils.utils import accuracy, AverageMeter


class AncestralModel(l.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.total_loss = AverageMeter()
        self.total_acc1 = AverageMeter()
        self.total_acc5 = AverageMeter()

        self.global_attention = AttentionModule(self.cfg["ninp"], self.cfg["nhead"], self.cfg["nhid"])

        self.encoder0 = nn.Embedding(self.cfg["ntoken"], 56)
        self.encoder1 = nn.Embedding(self.cfg["nlevel"],  4)
        self.encoder2 = nn.Embedding(self.cfg["noctant"], 4)

        self.pos_emb = PosEmb(16)

        self.decoder = nn.Sequential(
            nn.Linear(self.cfg["ninp"], self.cfg["nhid"]),
            nn.GELU(),
            nn.Linear(self.cfg["nhid"], self.cfg["nout"]),
        )

        self.save_hyperparameters(logger=False)

    def forward(self, source):
        l, b, _, _ = source.size()

        occupy = source[:, :,  :, 0]
        level  = source[:, :,  :, 1]
        octant = source[:, :,  :, 2]
        laser  = source[:, :,  :, 6]
        phi    = source[:, :,  :, 7]
        pos    = source[:, :, -1, 3:6].float()

        # feature embedding
        occupy_emb = self.encoder0(occupy)
        level_emb  = self.encoder1(level)
        octant_emb = self.encoder2(octant)

        # local geometric feature extraction
        parent_pos_emb = self.pos_emb(occupy, level, octant, laser, phi, pos)

        emb = torch.cat([occupy_emb, level_emb, octant_emb], dim=-1).reshape(l, b, -1)
        emb = torch.cat([emb, parent_pos_emb], dim=-1)

        output = self.global_attention(emb, mask=None)  # (l, b, c)

        output = self.decoder(output)

        return output

    def process_batch_data(self, batch):
        batch = batch.permute(1, 0, 2, 3)

        target = batch[:, :, -1, 0]

        target_1 = target[0::2].clone()
        target_2 = target[1::2].clone()

        source_1 = batch.clone()
        source_2 = batch.clone()

        source_1[   :, :, -1, 0] = 255
        source_2[1::2, :, -1, 0] = 255

        return source_1, source_2, target_1, target_2, target

    def training_step(self, batch, batch_idx):
        source_1, source_2, target_1, target_2, target = self.process_batch_data(batch)

        output_1 = self.forward(source_1)
        output_2 = self.forward(source_2)

        output_1 = output_1[0::2]
        output_2 = output_2[1::2]

        output = torch.zeros([self.cfg["ctx_win"], self.cfg["batch_size"], self.cfg["nout"]], device=batch.device)
        output[0::2] = output_1
        output[1::2] = output_2

        output   = output.reshape(-1, self.cfg["nout"])
        output_1 = output_1.reshape(-1, self.cfg["nout"])
        output_2 = output_2.reshape(-1, self.cfg["nout"])

        target   = target.reshape(-1).long()
        target_1 = target_1.reshape(-1).long()
        target_2 = target_2.reshape(-1).long()

        criterion = nn.CrossEntropyLoss(label_smoothing=0.)

        loss_1 = criterion(output_1, target_1) / math.log(2)
        loss_2 = criterion(output_2, target_2) / math.log(2)

        loss = loss_1 + loss_2

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        self.total_loss.update(loss.item())
        self.total_acc1.update(acc1.item())
        self.total_acc5.update(acc5.item())

        if batch_idx % self.cfg["log_on_bar_interval"] == 0:
            self.log("loss", self.total_loss.avg, prog_bar=True)
            self.log("acc1", self.total_acc1.avg, prog_bar=True)
            self.log("acc5", self.total_acc5.avg, prog_bar=True)
            self.log("lr",   self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)

            self.total_loss.reset()
            self.total_acc1.reset()
            self.total_acc5.reset()

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg["lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

        return [optimizer], [scheduler]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        if self.trainer.global_step < 2500:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / 5000.0 + 0.5)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.cfg["lr"]
        else:
            for pg in optimizer.param_groups:
                pg["lr"] = max(self.cfg["lr"] / 10, pg["lr"])

        optimizer.step(closure=optimizer_closure)


    def load_pretrained(self, path):
        state_dict_pre = torch.load(path, map_location="cuda")["state_dict"]
        state_dict_ref = self.state_dict()

        keys = list(state_dict_pre.keys())

        for key in keys:
            if '_orig_mod.' in key:
                del_key = key.replace('_orig_mod.', '')
                state_dict_pre[del_key] = state_dict_pre[key]
                del state_dict_pre[key]

        keys = list(state_dict_pre.keys())

        for k in keys:
            if k not in state_dict_ref or state_dict_pre[k].shape != state_dict_ref[k].shape:
                state_dict_pre.pop(k)

        torch.save(state_dict_pre, "static_nuscenes.ckpt")
        return self.load_state_dict(state_dict_pre, strict=True)
