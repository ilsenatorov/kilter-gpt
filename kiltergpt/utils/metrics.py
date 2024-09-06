from typing import Iterable

import torch


def get_histogram(tensor_list: Iterable[torch.Tensor], vocab_size: int, normalize: bool = True) -> torch.Tensor:
    """Calculate the token distribution of a list of long tensors."""
    hist = torch.zeros(vocab_size)
    for t in tensor_list:
        if len(t.size()) == 2:
            t = t.view(-1)
        hist += torch.bincount(t, minlength=vocab_size)
    if normalize:
        hist /= hist.sum()
    return hist


def jaccard_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a and b are of shape (samples, features). Both tensors should be binary."""
    intersection = (a * b).sum(dim=1)
    union = (a + b).sum(dim=1) - intersection
    return intersection / union
