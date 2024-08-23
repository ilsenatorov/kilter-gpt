import math
import random
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .tokenizer import Tokenizer


class KilterDataset(Dataset):
    def __init__(
        self,
        filename: str | Path,
        tokenizer: Tokenizer,
        *,
        context_len: int = 64,  # 1 hold == 2 tokens
        shuffle_tokens: bool = True,
        label_smoothing: bool = True,
        prompt_size: float = 0.2,
        min_tokens: int = 5,  # Always have at least 5 tokens - BOS, ANGLE, GRADE, HOLD1, COLOR1
        subset: float = 1.0,
    ):
        assert 0 < subset <= 1, f"Subset must be between 0 and 1, got {subset}"
        self.df = pd.read_csv(filename).sample(frac=subset)
        self.tokenizer = tokenizer
        self.context_len = context_len
        self.shuffle_tokens = shuffle_tokens
        self.label_smoothing = label_smoothing
        self.prompt_size = prompt_size
        self.min_tokens = min_tokens
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
        prompt_size = max(math.ceil(n_tokens * self.prompt_size), self.min_tokens)
        x = self.tokenizer.pad(tokenized[:prompt_size], self.context_len)
        y = self.tokenizer.pad(tokenized, self.context_len)
        return x, y

    def __getitem__(self, idx: int) -> tuple[torch.LongTensor, torch.Tensor]:
        if self.eval:
            return self._get_item_eval(idx)
        else:
            return self._get_item_train(idx)

    def _get_item_train(self, idx: int) -> tuple[torch.LongTensor, torch.Tensor]:
        """Get a random contiguous sequence of tokens from the frames column. Pad left to context_len."""
        row = self.df.iloc[idx]
        frames = row["frames"]
        tokenized = self.tokenizer.encode(
            frames,
            row["angle"].item(),
            row["font_grade"],
            shuffle=self.shuffle_tokens,
        )
        n = tokenized.size(0)  # total tokens
        if n <= self.min_tokens:
            end = n
        else:
            end = random.randint(self.min_tokens, n)
        start = max(0, end - self.context_len - 1)  # buffer start, - 1 for buffer overlap
        buffer = tokenized[start:end]
        x = buffer[:-1]
        y = buffer[1:]
        if self.label_smoothing:
            y = self._create_smoothed_labels(y)
        x = self.tokenizer.pad(x, self.context_len)
        y = self.tokenizer.pad(y, self.context_len)
        return x, y

    def _create_smoothed_labels(self, y: torch.LongTensor) -> torch.FloatTensor:
        """Make all holds equally valid."""
        nopad = y[y != self.tokenizer.pad_token_id]
        labels = torch.nn.functional.one_hot(nopad, num_classes=self.tokenizer.vocab_size).float()
        # Positions of hold tokens in the y tensor
        hold_indices = torch.tensor([x for x in range(2, nopad.size(0), 2)])
        hold_values = nopad[hold_indices]
        for idx, i in enumerate(hold_indices):
            labels[i, hold_values[idx:]] = 1
        return labels
