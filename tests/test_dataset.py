import pandas as pd
import pytest
import torch

from kiltergpt.data.datasets import KilterGPTDataset
from kiltergpt.data.tokenizer import Tokenizer


@pytest.fixture
def tokenizer():
    return Tokenizer.from_json("data/tokenizer.json")


@pytest.fixture
def dataset(tokenizer):
    # Create a sample dataframe
    data = {"frames": ["p1234r12p1211r13p1333r13p1421r15", "p1200r14"], "angle": [10, 20], "font_grade": ["7a", "7b"]}
    df = pd.DataFrame(data)
    df.to_csv("data/sample.csv", index=False)
    return KilterGPTDataset("data/sample.csv", tokenizer)


def test_dataset_length(dataset):
    assert len(dataset) == 2


def test_data_generation_consistency(dataset):
    x, y = dataset[0]
    assert x.size(0) == dataset.context_len
    assert y.size(0) == dataset.context_len


def test_evaluation_mode(dataset):
    dataset.eval = True
    x, y = dataset[0]
    assert x.size(0) == dataset.context_len
    assert y.size(0) == dataset.context_len


def test_label_smoothing(dataset):
    dataset.label_smoothing = True
    _, y = dataset[0]
    assert y.size(0) == dataset.context_len
    assert y.size(1) == dataset.tokenizer.vocab_size
    assert y.dtype == torch.float32
    nopad = y[y != dataset.tokenizer.pad_token_id]
    assert (nopad[torch.isin(nopad, dataset.tokenizer.hold_token_ids)][:-1] > 1).all()


if __name__ == "__main__":
    pytest.main()
