from argparse import ArgumentParser, Namespace

import lightning as L
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from src.data.datasets import KilterGPTDataset
from src.models.gpt import GPTModel
from src.utils import str_to_bool

L.seed_everything(42)
torch.set_float32_matmul_precision("high")


parser = ArgumentParser()
parser.add_argument("--batch_size", type=int, default=512)
parser.add_argument("--epochs", type=int, default=250)
parser.add_argument("--vocab_size", type=int, default=600)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--wd", type=float, default=1e-5)
parser.add_argument("--n_embed", type=int, default=512)
parser.add_argument("--num_blocks", type=int, default=8)
parser.add_argument("--num_heads", type=int, default=8)
parser.add_argument("--head_size", type=int, default=512 // 8)
parser.add_argument("--context_len", type=int, default=64)
parser.add_argument("--attn_drop_value", type=float, default=0.2)
parser.add_argument("--multihead_drop_value", type=float, default=0.2)
parser.add_argument("--ffn_drop_value", type=float, default=0.2)
parser.add_argument("--min_tokens", type=int, default=10)
parser.add_argument("--angle", type=str_to_bool, default=True)
parser.add_argument("--grade", type=str_to_bool, default=True)
config = parser.parse_args()


ds = KilterGPTDataset(
    "data/raw/climbs.csv",
    context_len=config.context_len,
    min_tokens=config.min_tokens,
    deduplicate=True,
    angle=config.angle,
    grade=config.grade,
)
train, val = random_split(ds, [0.8, 0.2])

train_dl = DataLoader(train, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=8)
val_dl = DataLoader(val, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=8)

model = GPTModel(config)

# prompts = [
#     ("p1185r12p1198r12", 40, "7b"),
#     ("p1128r12p1391r14", 45, "7a"),
#     ("p1315r13p1385r14", 30, "6a"),
#     ("p1157r15p1203r13p1223r13p1270r13", 30, "6b"),
# ]

trainer = Trainer(
    devices=-1,
    max_epochs=config.epochs,
    logger=[WandbLogger(project="kilter-gpt", config=config, log_model=True)],
    precision="bf16-mixed",
    callbacks=[
        L.pytorch.callbacks.EarlyStopping(monitor="val/loss", patience=10),
        L.pytorch.callbacks.ModelCheckpoint(monitor="val/loss", mode="min"),
    ],
)

trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
