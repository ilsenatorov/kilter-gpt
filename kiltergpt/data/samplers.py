import math
import random
from typing import Iterable

from torch.utils.data.sampler import BatchSampler

from .datasets import KilterDataset


class DynamicBatchSampler(BatchSampler):
    def __init__(self, dataset: KilterDataset, max_num_tokens: int, shuffle: bool = True):
        self.dataset = dataset
        self.max_num_tokens = max_num_tokens
        self.shuffle = shuffle

    def __iter__(self) -> Iterable[list[int]]:
        if self.shuffle:
            self.dataset.bucket_shuffle()
        batches = []
        batch = []
        num_tokens = 0
        for idx in range(len(self.dataset)):
            sample_length = self.dataset.df.length.iloc[idx]  # positional, like KilterDataset.__getitem__
            if batch and num_tokens + sample_length > self.max_num_tokens:
                batches.append(batch)
                batch = []
                num_tokens = 0
            batch.append(idx)
            num_tokens += sample_length

        if len(batch) > 0:
            batches.append(batch)
        if self.shuffle:  # buckets are contiguous, so without this an epoch walks one climb length at a time
            random.shuffle(batches)
        yield from batches

    def __len__(self):
        return math.ceil(self.dataset.df.length.sum() / self.max_num_tokens)
