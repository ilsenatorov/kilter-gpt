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
    ):
        self.df = pd.read_csv(filename)
        if deduplicate:
            self.df = self.df.drop_duplicates(subset=["frames"])
        self.tokenizer = self._get_tokenizer()
        self.context_len = context_len
        self.min_tokens = min_tokens

    def _get_tokenizer(self):
        return Tokenizer(self.df["frames"])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """Get a random contiguous sequence of tokens from the frames column. Pad left to context_len."""
        t = self.tokenizer.encode(shuffle_holds(self.df.iloc[idx]["frames"]))
        end = random.randint(self.min_tokens, t.size(0))
        start = max(0, end - self.context_len - 1)  # buffer start
        buffer = t[start:end]
        x = buffer[:-1]
        y = buffer[1:]
        x = pad_to(x, self.context_len, self.tokenizer.encode_map[self.tokenizer.pad_token])
        y = pad_to(y, self.context_len, self.tokenizer.encode_map[self.tokenizer.pad_token])
        return x, y
