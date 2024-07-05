from dataclasses import dataclass

import lightning as L
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from src.data.datasets import KilterGPTDataset
from src.models.gpt import GPTModel

L.seed_everything(42)
torch.set_float32_matmul_precision("high")


@dataclass
class Config:
    batch_size = 512
    epochs = 250
    vocab_size = 600
    lr = 1e-3
    wd = 1e-5
    n_embed = 512
    num_blocks = 8
    num_heads = 8
    head_size = n_embed // num_heads
    context_len = 64
    attn_drop_value = 0.2
    multihead_drop_value = 0.2
    ffn_drop_value = 0.2
    min_tokens = 20
    angle = True
    grade = True


config = Config()

ds = KilterGPTDataset(
    "data/raw/climbs.csv",
    context_len=config.context_len,
    min_tokens=config.min_tokens,
    deduplicate=False,
    angle=config.angle,
    grade=config.grade,
)
train, val = random_split(ds, [0.8, 0.2])

train_dl = DataLoader(train, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=8)
val_dl = DataLoader(val, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=8)

model = GPTModel(config)

trainer = Trainer(
    devices=-1,
    max_epochs=config.epochs,
    logger=[WandbLogger(project="kilter-gpt", config=vars(config), log_model=True)],
    precision="bf16-mixed",
    callbacks=[
        L.pytorch.callbacks.EarlyStopping(monitor="val/loss", patience=30),
        L.pytorch.callbacks.ModelCheckpoint(monitor="val/loss", mode="min"),
    ],
)

trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
