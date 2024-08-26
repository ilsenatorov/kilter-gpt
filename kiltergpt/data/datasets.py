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
        shuffle_tokens: bool = True,
        prompt_size: float = 0.2,
        subset: float = 1.0,
    ):
        assert 0 < subset <= 1, f"Subset must be between 0 and 1, got {subset}"
        self.df = pd.read_csv(filename).sample(frac=subset)
        self.df["length"] = self.df["frames"].apply(lambda x: len(x) // 4 + 4)
        self.sorted_shuffle()
        self.tokenizer = tokenizer
        self.smooth_labels = smooth_labels
        self.shuffle_tokens = shuffle_tokens
        self.prompt_size = prompt_size
        self.eval = False

    def __len__(self) -> int:
        return len(self.df)

    def _get_item_eval(self, idx: int) -> tuple[torch.LongTensor, torch.LongTensor]:
        row = self.df.iloc[idx]
        frames = row["frames"]
        tokenized = self.tokenizer.encode(
            frames,
            row["angle"].item(),
            row["font_grade"],
            shuffle=self.shuffle_tokens,
        )
        n_tokens = tokenized.size(0)
        prompt_size = max(math.ceil(n_tokens * self.prompt_size), 5)
        return tokenized[:prompt_size], tokenized

    def __getitem__(self, idx: int) -> tuple[torch.LongTensor, torch.Tensor]:
        if self.eval:
            return self._get_item_eval(idx)
        else:
            return self._get_item_train(idx)

    def _get_item_train(self, idx: int) -> tuple[torch.LongTensor, torch.Tensor]:
        """Get a training item. This will return a tuple of two tensors, x and y, where x is the input and y is the target."""
        row = self.df.iloc[idx]
        frames = row["frames"]
        tokenized = self.tokenizer.encode(
            frames,
            row["angle"].item(),
            row["font_grade"],
            shuffle=self.shuffle_tokens,
        )
        x = tokenized[:-1]
        y = tokenized[1:]
        if self.smooth_labels:
            y = self.smooth_y(tokenized[1:])
        return x, y

    def smooth_y(self, y: torch.Tensor) -> torch.Tensor:
        smooth_y = F.one_hot(y, num_classes=self.tokenizer.vocab_size).to(torch.float32)
        hold_positions = torch.arange(2, y.size(0) - 1, 2)
        holds = y[hold_positions]
        for i in range(len(holds)):
            smooth_y[hold_positions[i], holds[i:]] = 1
        return smooth_y

    def sorted_shuffle(self):
        self.df = self.df.sample(frac=1).reset_index(drop=True)
        self.df = self.df.sort_values(by="length", ascending=True).reset_index(drop=True)

    def __repr__(self):
        return f"KilterDataset of length {self.__len__()}"
