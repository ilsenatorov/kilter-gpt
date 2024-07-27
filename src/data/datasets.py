import random

import pandas as pd
import torch
from torch.utils.data import Dataset

from .tokenizer import Tokenizer, pad_to


class KilterGPTDataset(Dataset):
    def __init__(
        self,
        filename,
        context_len: int = 64,  # 1 hold == 2 tokens
        min_tokens: int = 5,  # smallest number of tokens in a sequence
        shuffle_tokens: bool = True,
        label_smoothing: bool = True,
        prompt_size: float = 0.5,
    ):
        self.context_len = context_len
        self.min_tokens = min_tokens
        self.shuffle_tokens = shuffle_tokens
        self.df = pd.read_csv(filename)
        self.tokenizer = self._get_tokenizer()
        self.label_smoothing = label_smoothing
        self.eval = False
        self.prompt_size = prompt_size

    def _get_tokenizer(self):
        return Tokenizer.from_df(self.df)

    def __len__(self):
        return len(self.df)

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
            correct_set = self._get_correct_set(tokenized, end)
            y = self._create_smoothed_labels(y, correct_set)
        x = self.tokenizer.pad(x, self.context_len)
        y = self.tokenizer.pad(y, self.context_len)
        return x, y

    def _get_item_eval(self, idx: int):
        row = self.df.iloc[idx]
        frames = row["frames"]
        tokenized = self.tokenizer.encode(
            frames,
            row["angle"].item(),
            row["font_grade"],
            shuffle=self.shuffle_tokens,
        )
        n_tokens = tokenized.size(0)
        prompt_size = int(n_tokens * self.prompt_size)
        x = self.tokenizer.pad(tokenized[:prompt_size], self.context_len)
        y = self.tokenizer.pad(tokenized, self.context_len)
        return x, y

    def __getitem__(self, idx: int):
        if self.eval:
            return self._get_item_eval(idx)
        else:
            return self._get_item_train(idx)

    def _create_smoothed_labels(self, y: torch.LongTensor, correct_set: torch.Tensor) -> torch.FloatTensor:
        """If the last token is a hold, smooth labels for all remaining holds."""
        smooth_labels = torch.nn.functional.one_hot(y, num_classes=len(self.tokenizer.encode_map)).float()
        if y[-1].item() in self.tokenizer.hold_token_ids:
            if correct_set.size(0) > 0:
                smooth_labels[-1, correct_set] = 1.0
        return smooth_labels

    def _get_correct_set(self, tokenized: torch.LongTensor, end: int):
        """Get the set of correct tokens for the last token in the sequence."""
        suffix = tokenized[end:]
        return suffix[torch.isin(suffix, self.tokenizer.hold_token_ids)]
