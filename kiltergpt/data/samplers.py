import math

from torch.utils.data.sampler import BatchSampler

from .datasets import KilterDataset


class DynamicBatchSampler(BatchSampler):
    def __init__(self, dataset: KilterDataset, max_num_tokens: int, shuffle: bool = True):
        self.dataset = dataset
        self.max_num_tokens = max_num_tokens
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            self.dataset.sorted_shuffle()
        batch = []
        num_tokens = 0
        for idx in range(len(self.dataset)):
            sample_length = self.dataset.df.length[idx]
            if num_tokens + sample_length > self.max_num_tokens:
                yield batch
                batch = []
                num_tokens = 0
            batch.append(idx)
            num_tokens += sample_length

        if len(batch) > 0:
            yield batch

    def __len__(self):
        return math.ceil(self.dataset.df.length.sum() / self.max_num_tokens)
