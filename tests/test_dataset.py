import pandas as pd
import pytest
import torch

from kiltergpt.data.datamodules import KilterDataModule
from kiltergpt.data.datasets import KilterDataset
from kiltergpt.data.samplers import DynamicBatchSampler
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
    return KilterDataset(dataset_dir / "train.csv", Tokenizer())


def test_dataset_length(dataset):
    assert len(dataset) == 2


def test_data_generation_consistency(dataset):
    x, angle, grade, y = dataset[0]
    assert x.size(0) == y.size(0)
    assert (x[1:] == y[:-1]).all()


def test_evaluation_mode(dataset):
    dataset.eval = True
    x, angle, grade, y = dataset[0]
    # assert that all of x is in y
    assert (x == y[: x.size(0)]).all()


def test_datamodule(dataset_dir):
    datamodule = KilterDataModule(data_dir=dataset_dir, batch_size=2, num_workers=0)
    datamodule.setup()
    batch = next(iter(datamodule.train_dataloader()))
    assert len(batch) == 4
    assert datamodule.train.eval is False
    assert datamodule.val.eval is False
    assert datamodule.test.eval is True


def test_batch_sampler(dataset):
    sampler = DynamicBatchSampler(dataset, 100)
    for batch in sampler:
        assert sum(dataset.df.length[idx] for idx in batch) <= 100
    assert len(sampler) == 1
    sampler = DynamicBatchSampler(dataset, 13)
    for batch in sampler:
        assert sum(dataset.df.length[idx] for idx in batch) <= 13
    assert len(sampler) == 2
