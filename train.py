from argparse import ArgumentParser

import lightning as L
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

from src.data.datasets import KilterGPTDataset
from src.models.gpt import GPTModel
from src.utils import str_to_bool

L.seed_everything(42)
torch.set_float32_matmul_precision("medium")


parser = ArgumentParser()
# dataset params
parser.add_argument("--dataset", type=str, default="data/raw/climbs.csv")
parser.add_argument("--min_tokens", type=int, default=10, help="Minimum number of tokens in a climb")
parser.add_argument("--angle", type=str_to_bool, default=True)
parser.add_argument("--grade", type=str_to_bool, default=True)
parser.add_argument("--label_smoothing", type=str_to_bool, default=True)
parser.add_argument("--grade_mask_rate", type=float, default=0.0)
# training params
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--epochs", type=int, default=250)
parser.add_argument("--lr", type=float, default=6e-4)
parser.add_argument("--warmup_steps", type=float, default=6e-4)
parser.add_argument("--wd", type=float, default=1e-1)
# model params
parser.add_argument("--n_head", type=int, default=8)
parser.add_argument("--n_layer", type=int, default=8)
parser.add_argument("--n_embed", type=int, default=512)
parser.add_argument("--context_len", type=int, default=64)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--bias", type=str_to_bool, default=False)
config = parser.parse_args()

ds = KilterGPTDataset(
    config.dataset,
    context_len=config.context_len,
    min_tokens=config.min_tokens,
    angle=config.angle,
    grade=config.grade,
    grade_mask_rate=config.grade_mask_rate,
    label_smoothing=config.label_smoothing,
)

train, val = random_split(ds, [0.8, 0.2])

train_dl = DataLoader(train, batch_size=config.batch_size, shuffle=True, pin_memory=True, num_workers=16)
val_dl = DataLoader(val, batch_size=config.batch_size, shuffle=False, pin_memory=True, num_workers=16)

config.vocab_size = len(ds.tokenizer.encode_map)
config.total_steps = len(train_dl) * config.epochs
model = GPTModel(config, ds.tokenizer)

trainer = Trainer(
    devices=-1,
    max_epochs=config.epochs,
    logger=[WandbLogger(project="kilter-gpt", config=config, log_model=True)],
    precision="bf16-mixed",
    callbacks=[
        L.pytorch.callbacks.EarlyStopping(monitor="val/loss", patience=20),
        L.pytorch.callbacks.ModelCheckpoint(monitor="val/loss", mode="min"),
        # L.pytorch.callbacks.StochasticWeightAveraging(swa_lrs=1e-2),
        L.pytorch.callbacks.LearningRateMonitor(logging_interval="step"),
    ],
)

trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)
