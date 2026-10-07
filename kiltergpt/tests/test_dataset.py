from itertools import pairwise

import pandas as pd
import pytest
import torch
from torch.utils.data import RandomSampler

from kiltergpt.data.datamodules import KilterDataModule
from kiltergpt.data.datasets import (
    KilterDataset,
    climb_signature,
    climb_split,
    deduplicate_climbs,
    legacy_quality_scale,
    post_archive_holdout,
    restore_legacy_stats,
    split_by_climb,
)
from kiltergpt.data.samplers import DynamicBatchSampler
from kiltergpt.data.tokenizer import Tokenizer


@pytest.fixture
def tokenizer():
    return Tokenizer()


@pytest.fixture
def dataset_dir(tmp_path):
    # Create a sample dataframe
    frames = {split: [] for split in ("train", "val", "test")}
    for hold in Tokenizer.hold_tokens():
        candidate = f"p1200r12{hold}r14"
        if hold != "p1200":
            frames[climb_split(candidate)].append(candidate)
        if len(frames["train"]) >= 2 and frames["val"] and frames["test"]:
            break
    for split in frames:
        df = pd.DataFrame(
            {"frames": frames[split][: 2 if split == "train" else 1], "angle": 40, "difficulty_average": 25}
        )
        df.to_csv(tmp_path / f"{split}.csv", index=False)
    return tmp_path


@pytest.fixture
def dataset(dataset_dir):
    return KilterDataset(dataset_dir / "train.csv", Tokenizer())


def test_split_by_climb():
    rows = []
    for i in range(10):
        frames = f"p{1200 + i}r12p{1300 + i}r14"
        rows.extend(
            [
                {"uuid": f"{i}", "frames": frames, "angle": 30},
                {"uuid": f"{i}", "frames": frames, "angle": 40},
                {"uuid": f"copy-{i}", "frames": f"p{1300 + i}r14p{1200 + i}r12", "angle": 30},
            ]
        )
    df = pd.DataFrame(rows)
    splits = split_by_climb(df)
    groups = [{climb_signature(frames) for frames in group.frames} for group in splits]
    assert sum(map(len, groups)) == 10
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    ids = [set(group.uuid) for group in splits]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    for split, group in zip(("train", "val", "test"), splits, strict=True):
        assert all(climb_split(frames) == split for frames in group.frames)
    assert set(split_by_climb(df.iloc[:5])[0].uuid) == set(splits[0].uuid) & set(df.iloc[:5].uuid)


def test_deduplicate_climbs():
    df = pd.DataFrame(
        [
            {"uuid": "a", "frames": "p1200r12p1201r14", "angle": 40, "ascensionist_count": 0, "quality_average": 1},
            {"uuid": "b", "frames": "p1201r14p1200r12", "angle": 40, "ascensionist_count": 2, "quality_average": 4},
            {"uuid": "c", "frames": "p1200r12p1201r14", "angle": 45, "ascensionist_count": 0, "quality_average": 1},
        ]
    )
    result = deduplicate_climbs(df)
    assert set(result.uuid) == {"b", "c"}


def test_restore_legacy_stats():
    current = pd.DataFrame(
        [
            {
                "uuid": "a",
                "angle": 40,
                "frames": "p1200r12",
                "ascensionist_count": 0,
                "quality_average": 1,
                "difficulty_average": 26,
            },
            {
                "uuid": "b",
                "angle": 40,
                "frames": "p1201r12",
                "ascensionist_count": 0,
                "quality_average": 1,
                "difficulty_average": 26,
            },
            {
                "uuid": "c",
                "angle": 40,
                "frames": "p1200r12",
                "ascensionist_count": 5,
                "quality_average": 3,
                "difficulty_average": 25,
            },
        ]
    )
    old = pd.DataFrame(
        [
            {
                "uuid": uuid,
                "angle": 40,
                "frames": "p1200r12",
                "ascensionist_count": 10,
                "quality_average": 4,
                "difficulty_average": 22,
            }
            for uuid in ("a", "b", "c")
        ]
    )
    restored = restore_legacy_stats(current, old).set_index("uuid")
    assert restored.loc[
        "a", ["ascensionist_count", "quality_average", "difficulty_average", "stats_source"]
    ].tolist() == [10, 4, 22, "2024"]
    assert restored.loc["b", "stats_source"] == "2026"
    assert restored.loc["c", "stats_source"] == "2026"


def test_post_archive_holdout():
    legacy = pd.DataFrame([{"uuid": "a", "frames": "p1200r12p1201r14", "created_at": "2024-08-31 10:00:00"}])
    candidates = pd.DataFrame(
        [
            {"uuid": "a", "frames": "p1202r12p1203r14", "created_at": "2024-09-01T10:00:00Z"},
            {"uuid": "b", "frames": "p1201r14p1200r12", "created_at": "2025-01-01T10:00:00Z"},
            {"uuid": "c", "frames": "p1202r12p1203r14", "created_at": "2024-08-31T23:59:59Z"},
            {"uuid": "d", "frames": "p1202r12p1203r14", "created_at": "2025-01-01T10:00:00Z"},
        ]
    )
    assert post_archive_holdout(candidates, legacy).uuid.tolist() == ["d"]


def test_dataset_length(dataset):
    assert len(dataset) == 2


def test_post_archive_split_guard(dataset_dir):
    post_file = dataset_dir / "test-post-2024.csv"
    pd.read_csv(dataset_dir / "test.csv").to_csv(post_file, index=False)
    assert len(KilterDataset(post_file, Tokenizer())) == 1
    pd.read_csv(dataset_dir / "train.csv").to_csv(post_file, index=False)
    with pytest.raises(ValueError, match="outside the frozen test split"):
        KilterDataset(post_file, Tokenizer())


def test_frozen_split_rejects_wrong_csv(dataset_dir):
    wrong = pd.read_csv(dataset_dir / "test.csv")
    wrong.to_csv(dataset_dir / "train.csv", index=False)
    with pytest.raises(ValueError, match="outside the frozen train split"):
        KilterDataset(dataset_dir / "train.csv", Tokenizer(), subset=0.1)


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
    assert isinstance(datamodule.train_dataloader().sampler, RandomSampler)  # CSVs are sorted by ascents
    assert datamodule.train.eval is False
    assert datamodule.val.eval is False
    assert datamodule.test.eval is True
    for split in (datamodule.val, datamodule.test):
        first = split[0]
        assert all(torch.equal(a, b) for a, b in zip(first, split[0], strict=True))
        assert split.shuffle_holds is False


def test_batch_sampler(dataset):
    sampler = DynamicBatchSampler(dataset, 100)
    for batch in sampler:
        assert sum(dataset.df.length.iloc[idx] for idx in batch) <= 100
    assert len(sampler) == 1
    sampler = DynamicBatchSampler(dataset, 13)
    for batch in sampler:
        assert sum(dataset.df.length.iloc[idx] for idx in batch) <= 13
    assert len(sampler) == 2


def test_batch_sampler_after_bucket_shuffle(tmp_path, tokenizer):
    holds = Tokenizer.hold_tokens()
    frames = [
        "p1100r12" + "".join(f"{h}r13" for h in holds[1 : 1 + n]) + "p1500r14" for n in range(2, 22) for _ in range(20)
    ]
    pd.DataFrame({"frames": frames, "angle": 40, "difficulty_average": 20}).to_csv(tmp_path / "mixed.csv", index=False)
    dataset = KilterDataset(tmp_path / "mixed.csv", tokenizer)  # bucket shuffle permutes index labels
    batches = list(DynamicBatchSampler(dataset, 40, shuffle=True))
    for batch in batches:
        assert len(batch) == 1 or sum(dataset.df.length.iloc[idx] for idx in batch) <= 40
    lengths = [dataset.df.length.iloc[batch[0]] for batch in batches]
    runs = sum(a != b for a, b in pairwise(lengths)) + 1
    assert runs > len(set(lengths))  # batches of one length are spread out, not walked bucket by bucket


def test_legacy_quality_scale():
    old = pd.Series([1.0, 1.5, 2.0, 2.5, 3.0] * 20)
    new = old * 2 - 1 + pd.Series([0.0, 0.0, 0.0, 0.0, 2.0] * 20)  # 1-5 scale, top bucket stretched
    to_legacy = legacy_quality_scale(new, old)
    assert to_legacy(new.iloc[:5]).tolist() == pytest.approx([1.0, 1.5, 2.0, 2.5, 3.0])
    assert to_legacy(pd.Series([0.0, 99.0])).tolist() == [1.0, 3.0]  # clipped to the legacy range
