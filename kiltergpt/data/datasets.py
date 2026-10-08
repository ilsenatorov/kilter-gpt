import hashlib
import math
from pathlib import Path
from typing import Callable

import numpy as np
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


def legacy_quality_scale(new: pd.Series, old: pd.Series) -> Callable[[pd.Series], np.ndarray]:
    """Quantile map from snapshot quality to the legacy 1-3 scale, fit on climbs rated in both.
    The snapshot mixes the legacy 3-star and the newer 5-star ratings inside each average, so there is no exact
    per-climb conversion; matching quantiles makes a quality threshold drop the same share of climbs in both."""
    pairs = pd.DataFrame({"new": new, "old": old}).dropna()
    qs = np.linspace(0, 1, 101)
    quantiles = pd.Series(np.quantile(pairs["old"], qs)).groupby(np.quantile(pairs["new"], qs)).mean()
    return lambda quality: np.interp(quality, quantiles.index, quantiles.to_numpy())


def restore_legacy_stats(df: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(old, on=["uuid", "angle"], how="left", suffixes=("", "_old"))
    same_climb = df["frames"].eq(df["frames_old"])
    rated = df["ascensionist_count"].gt(0)
    to_legacy = legacy_quality_scale(
        df.loc[rated & same_climb & df["ascensionist_count_old"].gt(0), "quality_average"],
        df.loc[rated & same_climb & df["ascensionist_count_old"].gt(0), "quality_average_old"],
    )
    df.loc[rated, "quality_average"] = to_legacy(df.loc[rated, "quality_average"])
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
        popularity: bool = False,
    ):
        """popularity: also return each climb's ascent count on Tokenizer.popularity's scale, as a fifth item"""
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
        if smooth_labels and tokenizer.role_first:
            raise ValueError("smooth_labels assumes hold-first tokens")
        self.tokenizer = tokenizer
        self.smooth_labels = smooth_labels
        self.prompt_size = prompt_size
        self.popularity = popularity
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

    def _get_item_eval(self, idx: int) -> tuple[torch.Tensor, ...]:  # popularity adds a fifth item
        tokenized, angle, grade = self._get_whole_buffer(idx)
        n_tokens = tokenized.size(0)
        prompt_size = max(math.ceil(n_tokens * self.prompt_size), 5)
        return self._with_popularity(idx, (tokenized[:prompt_size], angle, grade, tokenized))

    def _with_popularity(self, idx: int, item: tuple) -> tuple:
        if not self.popularity:
            return item
        ascents = self.df["ascensionist_count"].iloc[idx]
        return item + (torch.tensor(self.tokenizer.popularity(ascents), dtype=torch.float32),)

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
        return self._with_popularity(idx, (x, angle, grade, y))

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
