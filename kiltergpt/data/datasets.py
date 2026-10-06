import hashlib
import math
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .tokenizer import Tokenizer


def climb_signature(frames: str) -> str:
    return "|".join(sorted(frames.split("p")[1:]))


def climb_split(frames: str) -> str:
    bucket = int.from_bytes(hashlib.sha256(climb_signature(frames).encode()).digest()[:8], "big") % 100
    return "train" if bucket < 90 else "val" if bucket < 99 else "test"


def restore_legacy_stats(df: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(old, on=["uuid", "angle"], how="left", suffixes=("", "_old"))
    restored = df["ascensionist_count"].eq(0) & df["ascensionist_count_old"].gt(0) & df["frames"].eq(df["frames_old"])
    df["stats_source"] = "2026"
    df.loc[restored, "stats_source"] = "2024"
    for column in ("ascensionist_count", "quality_average", "difficulty_average"):
        df.loc[restored, column] = df.loc[restored, f"{column}_old"]
    return df.drop(columns=["frames_old", "ascensionist_count_old", "quality_average_old", "difficulty_average_old"])


def post_archive_holdout(test: pd.DataFrame, legacy_climbs: pd.DataFrame) -> pd.DataFrame:
    old_signatures = {climb_signature(frames) for frames in legacy_climbs["frames"]}
    after_archive = test["created_at"].str[:10] > legacy_climbs["created_at"].max()[:10]
    unseen = ~test["frames"].map(climb_signature).isin(old_signatures)
    return test[after_archive & unseen & ~test["uuid"].isin(legacy_climbs["uuid"])]


def deduplicate_climbs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.assign(signature=df["frames"].map(climb_signature))
    df = df.sort_values(
        ["ascensionist_count", "quality_average", "uuid"], ascending=[False, False, True], kind="stable"
    )
    return df.drop_duplicates(["signature", "angle"]).drop(columns="signature")


def split_by_climb(df: pd.DataFrame):
    if (df.groupby("uuid")["frames"].nunique() > 1).any():
        raise ValueError("A climb UUID has multiple frame definitions")
    splits = df["frames"].map(climb_split)
    return df[splits == "train"], df[splits == "val"], df[splits == "test"]


class KilterDataset(Dataset):
    def __init__(
        self,
        filename: str | Path,
        tokenizer: Tokenizer,
        *,
        smooth_labels: bool = False,
        prompt_size: float = 0.2,
        subset: float = 1.0,
        shuffle_holds: bool = True,
    ):
        assert 0 < subset <= 1, f"Subset must be between 0 and 1, got {subset}"
        df = pd.read_csv(filename)
        expected_split = Path(filename).stem.split("-")[0]
        if expected_split in ("train", "val", "test") and not df["frames"].map(climb_split).eq(expected_split).all():
            raise ValueError(f"{filename} has climbs outside the frozen {expected_split} split; regenerate its CSVs")
        self.df = df.sample(frac=subset, random_state=42)
        self.df["length"] = self.df["frames"].apply(lambda x: len(x) // 4 + 4)
        self.shuffle_holds = shuffle_holds
        if shuffle_holds:
            self.bucket_shuffle()
        else:
            self.len_sort()
        self.tokenizer = tokenizer
        self.smooth_labels = smooth_labels
        self.prompt_size = prompt_size
        self.eval = False

    def __len__(self) -> int:
        return len(self.df)

    def _get_whole_buffer(self, idx: int):
        row = self.df.iloc[idx]
        tokenized, angle, grade = self.tokenizer.encode(
            row["frames"],
            row["angle"].item(),
            row["difficulty_average"],
            shuffle=self.shuffle_holds,
        )
        return tokenized, angle, grade

    def _get_item_eval(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tokenized, angle, grade = self._get_whole_buffer(idx)
        n_tokens = tokenized.size(0)
        prompt_size = max(math.ceil(n_tokens * self.prompt_size), 5)
        return tokenized[:prompt_size], angle, grade, tokenized

    def __getitem__(self, idx: int):
        if self.eval:
            return self._get_item_eval(idx)
        else:
            return self._get_item_train(idx)

    def _get_item_train(self, idx: int):
        tokenized, angle, grade = self._get_whole_buffer(idx)
        x = tokenized[:-1]
        y = tokenized[1:]
        if self.smooth_labels:
            y = self.smooth_y(y)
        return x, angle, grade, y

    def smooth_y(self, y: torch.Tensor) -> torch.Tensor:
        smooth_y = F.one_hot(y, num_classes=self.tokenizer.vocab_size).to(torch.float32)
        hold_positions = torch.arange(2, y.size(0) - 1, 2)
        holds = y[hold_positions]
        for i in range(len(holds)):
            smooth_y[hold_positions[i], holds[i:]] = 1
        return smooth_y

    def shuffle(self):
        """Just shuffle the dataframe"""
        self.df = self.df.sample(frac=1).reset_index(drop=True)

    def len_sort(self):
        """Sort the dataframe by length of frames"""
        self.df = self.df.sort_values(by="length", kind="stable").reset_index(drop=True)

    def bucket_shuffle(self):
        """Shuffle the bucket order and within the buckets.
        A bucket is all sequences of the same length."""
        self.shuffle()
        self.df = pd.concat(
            [group.sample(frac=1) for _, group in self.df.sample(frac=1).groupby("length", sort=False)]
        )

    def __repr__(self):
        return f"KilterDataset of length {self.__len__()}"
