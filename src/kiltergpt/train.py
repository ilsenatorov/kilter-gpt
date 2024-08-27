import lightning.pytorch as L
import torch

from .data import KilterDataModule
from .models import GPTModel


def train(config):
    L.seed_everything(42)
    torch.set_float32_matmul_precision("high")
    dm = KilterDataModule(
        batch_size=config.batch_size,
        max_num_tokens=config.max_num_tokens,
        subset=config.subset,
        num_workers=config.num_workers,
        smooth_labels=config.smooth_labels,
    )
    dm.setup()

    config.vocab_size = dm.vocab_size
    config.total_steps = len(dm.train_dataloader()) * config.epochs
    model = GPTModel(config, dm.tokenizer)

    trainer = L.Trainer(
        max_epochs=config.epochs,
        logger=[L.loggers.WandbLogger(project="kilter-gpt", config=config, log_model=True)],
        precision=config.precision,
        callbacks=[
            L.callbacks.EarlyStopping(monitor="val/loss", patience=40),
            L.callbacks.ModelCheckpoint(monitor="val/loss", mode="min"),
            L.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
        gradient_clip_algorithm="value",
        gradient_clip_val=1.0,
    )

    trainer.fit(model, datamodule=dm)
    if not config.only_train:
        trainer.test(model, datamodule=dm)
