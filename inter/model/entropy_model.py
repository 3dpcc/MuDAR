import time

import lightning as l
import torch
import yaml
from easydict import EasyDict

from model.group_model_2stage import AncestralModel

cfg = EasyDict(yaml.safe_load(open("config.yaml", "r")))

class EntropyModel(l.LightningModule):
    def __init__(self, path):
        super().__init__()

        self.sib_encoder = AncestralModel(cfg)

        self.sib_encoder.load_pretrained(path)

    def encode(self, prev, curr, level, elapsed):
        if level == 0:
            anc_src = curr.clone()
            anc_src[:, :, -1, 0] = 255

            start = time.time()
            output = self.sib_encoder(prev, anc_src) # (l, b, c)
            elapsed += time.time() - start
        else:
            anc_src = curr.clone()
            anc_src[:, :, -1, 0] = 255

            sib_src = curr.clone()
            sib_src[1::2, :, -1, 0] = 255

            start = time.time()
            output_1 = self.sib_encoder(prev, anc_src) # (l, b, c)
            elapsed += time.time() - start

            start = time.time()
            output_2 = self.sib_encoder(prev, sib_src) # (l, b, c)
            elapsed += time.time() - start

            output = torch.empty_like(output_2)
            output[0::2] = output_1[0::2]
            output[1::2] = output_2[1::2]

        return output.detach().cpu(), elapsed

    def decode(self, prev, curr, elapsed):
        start = time.time()
        output = self.sib_encoder(prev, curr)
        elapsed += time.time() - start

        return output.detach().cpu(), elapsed
