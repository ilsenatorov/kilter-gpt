from typing import Iterable

import torch


def get_histogram(tensor_list: Iterable[torch.Tensor], vocab_size: int, normalize: bool = True) -> torch.Tensor:
    hist = torch.zeros(vocab_size)
    for t in tensor_list:
        hist += torch.bincount(t, minlength=vocab_size)
    if normalize:
        hist /= hist.sum()
    return hist
