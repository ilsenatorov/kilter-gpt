import random

import pandas as pd
import torch
from torch.utils.data import Dataset

from ..utils import Tokenizer, pad_to, shuffle_holds


class KilterGPTDataset(Dataset):
    def __init__(
        self,
        filename,
        context_len: int = 32,  # 1 hold == 2 tokens
        min_tokens: int = 5,  # smallest number of tokens in a sequence
        shuffle_tokens: bool = True,
        angle: bool = False,
        grade: bool = False,
        grade_mask_rate: float = 0.0,
        label_smoothing: bool = False,
    ):
        self.context_len = context_len
        self.min_tokens = min_tokens
        self.angle = angle
        self.grade = grade
        self.shuffle_tokens = shuffle_tokens
        self.df = pd.read_csv(filename)
        self.tokenizer = self._get_tokenizer()
        self.grade_mask_rate = grade_mask_rate
        self.label_smoothing = label_smoothing
        if grade is False and grade_mask_rate > 0:
            raise ValueError("grade_mask_rate > 0 requires grade=True")

    def _get_tokenizer(self):
        return Tokenizer.from_df(self.df, angle=self.angle, grade=self.grade)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.LongTensor, torch.Tensor]:
        """Get a random contiguous sequence of tokens from the frames column. Pad left to context_len."""
        row = self.df.iloc[idx]
        frames = row["frames"]
        tokenized = self.tokenizer.encode(
            shuffle_holds(frames),
            row["angle"].item(),
            row["font_grade"],
            shuffle=self.shuffle_tokens,
        )
        n = tokenized.size(0)
        if n <= self.min_tokens:
            end = n
        else:
            end = random.randint(self.min_tokens, n)
        start = max(0, end - self.context_len - 1)  # buffer start
        buffer = tokenized[start:end]
        x = buffer[:-1]
        y = buffer[1:]
        x = pad_to(x, self.context_len, self.tokenizer.encode_map[self.tokenizer.pad_token])
        y = pad_to(y, self.context_len, self.tokenizer.encode_map[self.tokenizer.pad_token])
        if self.label_smoothing:
            correct_set = self._get_correct_set(tokenized, end)
            y = self._create_smoothed_labels(y, correct_set)
        if self.grade_mask_rate > 0:
            x = self.mask_grade(x)
        return x, y

    def _create_smoothed_labels(self, y: torch.LongTensor, correct_set) -> torch.FloatTensor:
        """If the last token is a hold, smooth labels for all remaining holds."""
        smoothed_labels = torch.zeros((self.context_len, len(self.tokenizer.encode_map)), dtype=torch.float32)
        smoothed_labels[torch.arange(self.context_len), y] = 1.0
        if y[-1].item() in self.tokenizer.hold_token_ids:
            if correct_set.size(0) > 0:
                smoothed_labels[-1, correct_set] = 1.0
        return smoothed_labels

    def _get_correct_set(self, tokenized: torch.LongTensor, end: int):
        """Get the set of correct tokens for the last token in the sequence."""
        suffix = tokenized[end:]
        return suffix[torch.isin(suffix, self.tokenizer.hold_token_ids)]

    def mask_grade(self, x: torch.LongTensor) -> torch.LongTensor:
        """Randomly mask the grade token."""
        mask = random.random() < self.grade_mask_rate
        if mask:
            x[torch.isin(x, self.tokenizer.grade_token_ids)] = self.tokenizer.mask_token_id
        return x
