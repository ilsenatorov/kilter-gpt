import pandas as pd
import pytest
import torch

from kiltergpt.data.datamodules import KilterDataModule
from kiltergpt.data.datasets import KilterGPTDataset
from kiltergpt.data.tokenizer import Tokenizer


@pytest.fixture
def tokenizer():
    return Tokenizer()


@pytest.fixture
def dataset_dir(tmp_path):
    # Create a sample dataframe
    data = {"frames": ["p1234r12p1211r13p1333r13p1421r15", "p1200r14"], "angle": [10, 20], "font_grade": ["7a", "7b"]}
    df = pd.DataFrame(data)
    for i in ["train", "val", "test"]:
        df.to_csv(tmp_path / f"{i}.csv", index=False)
    return tmp_path


@pytest.fixture
def dataset(dataset_dir):
    return KilterGPTDataset(dataset_dir / "train.csv", Tokenizer(), context_len=64)


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
    # FIXME fix this test part
    # nopad = y[y != dataset.tokenizer.pad_token_id]
    # assert (nopad[torch.isin(nopad, dataset.tokenizer.hold_token_ids())][:-1] > 1).all()


def test_datamodule(dataset_dir):
    datamodule = KilterDataModule(data_dir=dataset_dir, batch_size=2, num_workers=0)
    datamodule.setup()
    batch = next(iter(datamodule.train_dataloader()))
    assert len(batch) == 2
