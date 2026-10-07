from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import lightning.pytorch as L
import torch

from kiltergpt.data import KilterDataModule
from kiltergpt.models import GradePredictor

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--data_dir", type=str, default="data/processed-broad-2026", help="Same split as the GPT")
parser.add_argument("--min_ascents", type=int, default=1, help="Drop climbs whose grade is only the setter's")
parser.add_argument("--max_num_tokens", type=int, default=8192, help="Max number of tokens per batch")
parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for dataloader")
parser.add_argument("--subset", type=float, default=1.0, help="Fraction of datasets to take")
parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
parser.add_argument("--lr", type=float, default=3e-4, help="Max learning rate")
parser.add_argument("--wd", type=float, default=1e-2, help="Weight decay")
parser.add_argument("--n_head", type=int, default=4, help="Number of attention heads")
parser.add_argument("--n_layer", type=int, default=4, help="Number of transformer layers")
parser.add_argument("--n_embed", type=int, default=256, help="Embedding dimension")
parser.add_argument("--dropout", type=float, default=0.1, help="Dropout")
config = parser.parse_args()


class GradeDataModule(KilterDataModule):
    """Lightning re-runs setup() in fit/test, so the filtering lives here"""

    def setup(self, stage=None):
        super().setup(stage)
        self.test.eval = False  # whole climbs, not prompts
        for dataset in (self.train, self.val, self.test):
            dataset.df = dataset.df[dataset.df["ascensionist_count"] >= config.min_ascents]


L.seed_everything(42)
dm = GradeDataModule(
    data_dir=config.data_dir,
    max_num_tokens=config.max_num_tokens,
    num_workers=config.num_workers,
    subset=config.subset,
)
dm.setup()
config.total_steps = len(dm.train_dataloader()) * config.epochs

checkpoint = L.callbacks.ModelCheckpoint(monitor="val/mae", mode="min")
trainer = L.Trainer(
    max_epochs=config.epochs,
    logger=L.loggers.WandbLogger(project="kilter-gpt", job_type="grade-predictor", config=config),
    callbacks=[checkpoint, L.callbacks.EarlyStopping(monitor="val/mae", patience=5)],
    gradient_clip_val=1.0,
)
trainer.fit(GradePredictor(config), datamodule=dm)
with torch.serialization.safe_globals([type(config)]):
    trainer.test(ckpt_path="best", datamodule=dm)  # test/mae on real climbs = the floor for generated grade_mae
print(f"Best checkpoint: {checkpoint.best_model_path}")
