from pathlib import Path

import lightning as L
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
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
        prompt_size: float = 0.2,
        subset: float = 1.0,
    ):
        super().__init__()
        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prompt_size = prompt_size
        self.subset = subset

    def _get_dataset(self, csv_filename: str) -> KilterDataset:
        return KilterDataset(
            self.data_dir / csv_filename,
            self.tokenizer,
            prompt_size=self.prompt_size,
            subset=self.subset,
        )

    def collate_fn(self, batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = zip(*batch, strict=True)
        x = pad_sequence(x, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        y = pad_sequence(y, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        return x, y

    def setup(self, stage=None):
        self.tokenizer = Tokenizer()
        self.train = self._get_dataset("train.csv")
        self.val = self._get_dataset("val.csv")
        self.test = self._get_dataset("test.csv")
        self.test.eval = True
        self.vocab_size = self.tokenizer.vocab_size

    def _get_dataloader(self, dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train, shuffle=False)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.val)

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.test)

    def __repr__(self):
        if not hasattr(self, "train"):
            return "KilterDataModule"
        return f"KilterDataModule, train - {len(self.train)}, val - {len(self.val)}, test - {len(self.test)}"
