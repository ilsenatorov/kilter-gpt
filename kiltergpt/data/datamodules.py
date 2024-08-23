from pathlib import Path

import lightning as L
import pandas as pd
from torch.utils.data import DataLoader

from .datasets import KilterDataset
from .tokenizer import Tokenizer


class KilterDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str | Path = Path("data") / "processed",
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = True,
        context_len: int = 64,
        label_smoothing: bool = True,
        prompt_size: float = 0.2,
        min_tokens: int = 5,
    ):
        super().__init__()
        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.context_len = context_len
        self.label_smoothing = label_smoothing
        self.prompt_size = prompt_size
        self.min_tokens = min_tokens

    def setup(self, stage=None):
        self.tokenizer = Tokenizer()
        self.train = KilterDataset(
            self.data_dir / "train.csv",
            self.tokenizer,
            context_len=self.context_len,
            label_smoothing=self.label_smoothing,
            prompt_size=self.prompt_size,
            min_tokens=self.min_tokens,
        )

        self.val = KilterDataset(
            self.data_dir / "val.csv",
            self.tokenizer,
            context_len=self.context_len,
            label_smoothing=self.label_smoothing,
            prompt_size=self.prompt_size,
            min_tokens=self.min_tokens,
        )
        self.test = KilterDataset(
            self.data_dir / "test.csv",
            self.tokenizer,
            context_len=self.context_len,
            label_smoothing=self.label_smoothing,
            prompt_size=self.prompt_size,
            min_tokens=self.min_tokens,
        )
        self.test.eval = True
        self.vocab_size = self.tokenizer.vocab_size

    def _get_dataloader(self, dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=self.num_workers,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.val)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.test)
