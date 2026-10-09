from pathlib import Path

import lightning as L
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
        popularity (bool): Add each climb's ascent count (Tokenizer.popularity) as a fifth batch item.
        role_first (bool): Tokenize each hold as role then placement (r13 p1234) instead of placement then role.
        bottom_up (bool): Train on holds listed bottom to top (after a random prompt half the time), not shuffled.
        shuffle (bool): Shuffle the train set every epoch. The CSVs are sorted by ascensionist_count, so leave this on.
        test_subset (float): Random fraction of the test set, to make generation metrics cheaper in sweeps.
        test_file (str): Held-out CSV to use for testing.
        shuffle_test_prompts (bool): Prompt with a random subset of each test climb's holds, not the first ones in CSV
            order (sorted by placement, so mostly low bolt-ons, which biases generation away from screw-on holds).
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
        role_first: bool = False,
        bottom_up: bool = False,
        popularity: bool = False,
        shuffle: bool = True,
        test_file: str = "test.csv",
        test_subset: float = 1.0,
        shuffle_test_prompts: bool = False,
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
        self.role_first = role_first
        self.bottom_up = bottom_up
        self.popularity = popularity
        self.shuffle = shuffle
        self.test_file = test_file
        self.test_subset = test_subset
        self.shuffle_test_prompts = shuffle_test_prompts

    def _get_dataset(self, csv_filename: str) -> KilterDataset:
        return KilterDataset(
            self.data_dir / csv_filename,
            self.tokenizer,
            prompt_size=self.prompt_size,
            subset=self.subset * (self.test_subset if csv_filename == self.test_file else 1),
            smooth_labels=self.smooth_labels,
            shuffle_holds=csv_filename == "train.csv"
            or (csv_filename == self.test_file and self.shuffle_test_prompts),
            popularity=self.popularity,
        )

    def collate_fn(self, batch: list[tuple[torch.Tensor, ...]]) -> tuple[torch.Tensor, ...]:
        """(x, angle, grade, y), plus popularity (B, 1) when the datasets return it"""
        x, angle, grade, y, *popularity = zip(*batch, strict=True)
        x = pad_sequence(x, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        y = pad_sequence(y, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        angle = torch.tensor(angle, dtype=torch.float).unsqueeze(1)
        grade = torch.tensor(grade, dtype=torch.float).unsqueeze(1)
        return (x, angle, grade, y) + tuple(torch.tensor(p, dtype=torch.float).unsqueeze(1) for p in popularity)

    def setup(self, stage=None):
        self.tokenizer = Tokenizer(role_first=self.role_first, bottom_up=self.bottom_up)
        self.train = self._get_dataset("train.csv")
        self.val = self._get_dataset("val.csv")
        self.test = self._get_dataset(self.test_file)
        self.test.eval = True
        self.vocab_size = self.tokenizer.vocab_size

    def _get_dataloader(self, dataset, shuffle: bool = False) -> DataLoader:
        if self.max_num_tokens is None:
            return DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=shuffle,
                pin_memory=self.pin_memory,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
            )
        return DataLoader(
            dataset,
            batch_sampler=DynamicBatchSampler(dataset, self.max_num_tokens, shuffle=shuffle),
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train, shuffle=self.shuffle)

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.val, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=1,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

    def __repr__(self):
        if not hasattr(self, "train"):
            return "KilterDataModule"
        return f"KilterDataModule, train - {len(self.train)}, val - {len(self.val)}, test - {len(self.test)}"
