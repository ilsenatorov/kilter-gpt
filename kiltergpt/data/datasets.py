import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .tokenizer import Tokenizer


class KilterDataset(Dataset):
    def __init__(
        self,
        filename: str | Path,
        tokenizer: Tokenizer,
        *,
        smooth_labels: bool = False,
        prompt_size: float = 0.2,
        subset: float = 1.0,
    ):
        assert 0 < subset <= 1, f"Subset must be between 0 and 1, got {subset}"
        self.df = pd.read_csv(filename).sample(frac=subset)
        self.df["length"] = self.df["frames"].apply(lambda x: len(x) // 4 + 4)
        self.bucket_shuffle()
        self.tokenizer = tokenizer
        self.smooth_labels = smooth_labels
        self.prompt_size = prompt_size
        self.eval = False

    def __len__(self) -> int:
        return len(self.df)

    def _get_whole_buffer(self, idx: int):
        row = self.df.iloc[idx]
        tokenized, angle, grade = self.tokenizer.encode(
            row["frames"],
            row["angle"].item(),
            row["difficulty_average"],
            shuffle=True,
        )
        return tokenized, angle, grade

    def _get_item_eval(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokenized, angle, grade = self._get_whole_buffer(idx)
        n_tokens = tokenized.size(0)
        prompt_size = max(math.ceil(n_tokens * self.prompt_size), 5)
        return tokenized[:prompt_size], angle, grade, tokenized

    def __getitem__(self, idx: int):
        if self.eval:
            return self._get_item_eval(idx)
        else:
            return self._get_item_train(idx)

    def _get_item_train(self, idx: int):
        tokenized, angle, grade = self._get_whole_buffer(idx)
        x = tokenized[:-1]
        y = tokenized[1:]
        if self.smooth_labels:
            y = self.smooth_y(y)
        return x, angle, grade, y

    def smooth_y(self, y: torch.Tensor) -> torch.Tensor:
        smooth_y = F.one_hot(y, num_classes=self.tokenizer.vocab_size).to(torch.float32)
        hold_positions = torch.arange(2, y.size(0) - 1, 2)
        holds = y[hold_positions]
        for i in range(len(holds)):
            smooth_y[hold_positions[i], holds[i:]] = 1
        return smooth_y

    def shuffle(self):
        """Just shuffle the dataframe"""
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def len_sort(self):
        """Sort the dataframe by length of frames"""
        self.df = self.df.sort_values(by="length", ascending=True).reset_index(drop=True)

    def bucket_shuffle(self):
        """Shuffle the bucket order and within the buckets.
        A bucket is all sequences of the same length."""
        self.shuffle()
        self.df = pd.concat(
            [group.sample(frac=1) for _, group in self.df.sample(frac=1).groupby("length", sort=False)]
        )

    def __repr__(self):
        return f"KilterDataset of length {self.__len__()}"
