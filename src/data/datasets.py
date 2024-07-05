import random

import pandas as pd
from torch.utils.data import Dataset

from ..utils import Tokenizer, pad_to, shuffle_holds


class KilterGPTDataset(Dataset):
    def __init__(
        self,
        filename,
        context_len: int = 32,  # 1 hold == 2 tokens
        min_tokens: int = 5,  # smallest number of tokens in a sequence
        deduplicate: bool = True,
        angle: bool = False,
        grade: bool = False,
    ):
        self.context_len = context_len
        self.min_tokens = min_tokens
        self.angle = angle
        self.grade = grade
        self.df = pd.read_csv(filename)
        if deduplicate:
            self.df = self.df.drop_duplicates("frames")
        self.tokenizer = self._get_tokenizer()

    def _get_tokenizer(self):
        return Tokenizer.from_df(self.df, angle=self.angle, grade=self.grade)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Get a random contiguous sequence of tokens from the frames column. Pad left to context_len."""
        row = self.df.iloc[idx]
        tokenized = self.tokenizer.encode(shuffle_holds(row["frames"]), row["angle"].item(), row["font_grade"])
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
        return x, y
