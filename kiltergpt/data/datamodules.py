from pathlib import Path

import lightning as L
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .datasets import KilterDataset
from .samplers import DynamicBatchSampler
from .tokenizer import Tokenizer


class KilterDataModule(L.LightningDataModule):
    """Lightning DataModule for KilterGPT. Assumes that the data folder contains train.csv, val.csv, and test.csv.

    Args:

        data_dir (str | Path): Path to the data directory. Has to contain train.csv, val.csv, and test.csv.
        batch_size (int): Batch size to use.
        max_num_tokens (int, optional): If not None, uses DynamicBatchSampler to create batches with a maximum number of tokens.
        num_workers (int): Number of workers to use for loading data.
        pin_memory (bool): Whether to pin memory in DataLoader.
        prompt_size (float): Fraction of the sequence to use as prompt. Only used if the dataset is set to evaluation mode.
        subset (float): Fraction of the dataset to use. Useful for debugging.
    """

    def __init__(
        self,
        data_dir: str | Path = Path("data") / "processed",
        batch_size: int = 64,
        max_num_tokens: int | None = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        prompt_size: float = 0.2,
        subset: float = 1.0,
        smooth_labels: bool = False,
    ):
        super().__init__()
        if not isinstance(data_dir, Path):
            data_dir = Path(data_dir)
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.max_num_tokens = max_num_tokens
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prompt_size = prompt_size
        self.subset = subset
        self.smooth_labels = smooth_labels

    def _get_dataset(self, csv_filename: str) -> KilterDataset:
        return KilterDataset(
            self.data_dir / csv_filename,
            self.tokenizer,
            prompt_size=self.prompt_size,
            subset=self.subset,
            smooth_labels=self.smooth_labels,
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
        if self.max_num_tokens is None:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                pin_memory=True,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
        return DataLoader(
            dataset,
            batch_sampler=DynamicBatchSampler(dataset, self.max_num_tokens, shuffle=shuffle),
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
