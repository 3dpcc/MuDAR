import glob
import lightning.pytorch as pl
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint, RichModelSummary
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from easydict import EasyDict

from utils.dataset_node import NodeDataset
from model.group_model_2stage import AncestralModel
from utils.callbacks import ScriptBackupCallback

pl.seed_everything(1226, workers=True)

torch.set_float32_matmul_precision("high")

cfg = yaml.safe_load(open("config.yaml", "r"))
cfg = EasyDict(cfg)

train_set = NodeDataset(cfg.train_path)

train_loader = DataLoader(
    dataset=train_set,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_worker,
    drop_last=True,
    pin_memory=True
)

tensorboard_logger = TensorBoardLogger("./logs", name=cfg.exp_name)

checkpoint_callback = ModelCheckpoint(
    filename="model-{epoch:02d}-{step:06d}",
    every_n_train_steps=2000,
    save_top_k=-1,
)

file_list = glob.glob("model/*", recursive=True)
file_list = [f for f in file_list if f.endswith('py')]

script_backup_callback = ScriptBackupCallback(
    script_paths=[
        *file_list,
        "config.yaml",
        "train.py",
    ],
)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=[0],
    max_epochs=cfg.epoch,
    log_every_n_steps=cfg.log_interval,
    logger=tensorboard_logger,
    enable_model_summary=False,
    gradient_clip_val=1.0,
    gradient_clip_algorithm="norm",
    callbacks=[
        RichModelSummary(max_depth=1),
        checkpoint_callback,
        script_backup_callback,
    ],
)

model = AncestralModel(cfg)

if cfg.ckpt_path is not None:
    missing_keys, unexpected_keys = model.load_pretrained(cfg.ckpt_path)
    print(f"missing_keys: {missing_keys}")
    print(f"unexpected_keys: {unexpected_keys}")

trainer.fit(model, train_dataloaders=train_loader)
