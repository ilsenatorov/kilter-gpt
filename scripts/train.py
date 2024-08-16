from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import lightning as L
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger

from kiltergpt.data.datamodules import KilterDataModule
from kiltergpt.models.gpt import GPTModel
from kiltergpt.utils import str_to_bool

L.seed_everything(42)
torch.set_float32_matmul_precision("medium")


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
# dataset params
parser.add_argument("--label_smoothing", type=str_to_bool, default=True, help="Multiple choices for middle holds")
# training params
parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
parser.add_argument("--epochs", type=int, default=500, help="Number of epochs")
parser.add_argument("--lr", type=float, default=6e-4, help="Max learning rate")
parser.add_argument("--wd", type=float, default=1e-1, help="Weight decay")
# model params
parser.add_argument("--n_head", type=int, default=8, help="Number of attention heads")
parser.add_argument("--n_layer", type=int, default=8, help="Number of transformer layers")
parser.add_argument("--n_embed", type=int, default=512, help="Embedding dimension")
parser.add_argument("--context_len", type=int, default=64, help="Context length")
parser.add_argument("--dropout", type=float, default=0.2, help="Dropout")
parser.add_argument("--bias", type=str_to_bool, default=False, help="Use bias in attention layers")
config = parser.parse_args()

dm = KilterDataModule(
    batch_size=config.batch_size,
    context_len=config.context_len,
    label_smoothing=config.label_smoothing,
)
dm.setup()

config.vocab_size = dm.vocab_size
config.total_steps = len(dm.train_dataloader()) * config.epochs
model = GPTModel(config, dm.tokenizer)

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

trainer.fit(model, datamodule=dm)
# trainer.test(model, test_dataloaders=test_dl)
