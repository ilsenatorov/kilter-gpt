from typing import Iterable, Literal

import numpy as np
import pandas as pd
import torch


def shuffle_holds(climb: str) -> str:
    """Shuffle the holds in a climb"""
    holds = climb.split("p")[1:]
    np.random.shuffle(holds)
    return "".join(["p" + x.strip() for x in holds])


def pad_to(
    tensor: torch.Tensor,
    size: int,
    pad_value: int = 0,
    where: Literal["left", "right"] = "left",
) -> torch.Tensor:
    """Pad tensor to a specific size"""
    if where == "left":
        left_pad = size - tensor.size(0)
        right_pad = 0
    elif where == "right":
        left_pad = 0
        right_pad = size - tensor.size(0)
    pad = [left_pad, right_pad]
    if tensor.dim() == 2:
        pad = (0, 0, left_pad, right_pad)
    return torch.nn.functional.pad(tensor, pad, value=pad_value)


class Tokenizer:
    eos_token = "[EOS]"
    bos_token = "[BOS]"
    pad_token = "[PAD]"
    unk_token = "[UNK]"
    mask_token = "[MASK]"

    def __init__(self, encode_map: dict[str, int]):
        self.encode_map = encode_map
        self.decode_map = {v: k for k, v in self.encode_map.items()}
        self._set_special_token_ids()

    @property
    def angle_tokens(self):
        return [x for x in self.encode_map if x.startswith("a")]

    @property
    def hold_tokens(self):
        return [x for x in self.encode_map if x.startswith("p")]

    @property
    def hold_token_ids(self):
        return torch.tensor([self.encode_map[x] for x in self.hold_tokens])

    @staticmethod
    def color_tokens():
        return ["r12", "r13", "r14", "r15"]

    @staticmethod
    def special_tokens():
        return ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[MASK]"]

    @property
    def color_token_ids(self):
        return torch.tensor([self.encode_map[x] for x in self.color_tokens])

    @property
    def angle_token_ids(self):
        return torch.tensor([self.encode_map[x] for x in self.angle_tokens])

    @property
    def grade_tokens(self):
        return [x for x in self.encode_map if x.startswith("f")]

    @property
    def grade_token_ids(self):
        return torch.tensor([self.encode_map[x] for x in self.grade_tokens])

    @property
    def special_token_ids(self):
        return torch.tensor(
            [
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
                self.unk_token_id,
                self.mask_token_id,
            ]
        )

    def _set_special_token_ids(self):
        self.pad_token_id = self.encode_map[self.pad_token]
        self.bos_token_id = self.encode_map[self.bos_token]
        self.eos_token_id = self.encode_map[self.eos_token]
        self.unk_token_id = self.encode_map[self.unk_token]
        self.mask_token_id = self.encode_map[self.mask_token]

    @staticmethod
    def from_df(df: pd.DataFrame, angle: bool = True, grade: bool = True) -> "Tokenizer":
        hold_tokens, angle_tokens, grade_tokens = set(), set(), set()
        for frame in df["frames"].unique():
            for token in Tokenizer.split_tokens(frame):
                if token.startswith("p"):  # Add only hold tokens
                    hold_tokens.add(token)
        # Add angle and difficulty tokens
        if angle:
            for i in df["angle"].unique():
                angle_tokens.add(f"a{i}")
        if grade:
            for i in df["font_grade"].unique():
                grade_tokens.add(f"f{i}")
        tokens = (
            Tokenizer.special_tokens()
            + Tokenizer.color_tokens()
            + sorted(list(hold_tokens))
            + sorted(list(angle_tokens))
            + sorted(list(grade_tokens))
        )
        encode_map = {x: i for i, x in enumerate(tokens)}
        return Tokenizer(encode_map)

    @staticmethod
    def split_tokens(frames: str) -> list[str]:
        """Split the frames into tokens."""
        res = []
        for pair in frames.split("p")[1:]:
            hc = pair.split("r")
            if len(hc) == 1:
                res += [f"p{hc[0]}"]
            else:
                hold, color = hc
                res += [f"p{hold}", f"r{color}"]
        return res

    def encode(
        self,
        frames: str,
        angle: int = None,
        grade: str = None,
        *,
        shuffle: bool = False,
        bos: bool = True,
        eos: bool = True,
        pad: int = 0,
    ) -> torch.Tensor:
        tokens = []
        if bos:
            tokens.append(self.bos_token)
        if angle:
            tokens.append(f"a{angle}")
        if grade:
            tokens.append(f"f{grade}")
        if shuffle:
            frames = shuffle_holds(frames)
        tokens.extend(self.split_tokens(frames))
        if eos:
            tokens.append(self.eos_token)
        t = torch.tensor([self.encode_map[x] for x in tokens], dtype=torch.long)
        if pad:
            t = self.pad(t, pad)
        return t

    def onehot(self, frames: str) -> torch.Tensor:
        """Save presence/absence of each hold in a one-hot tensor"""
        t = torch.zeros(len(self.encode_map), dtype=torch.long)
        for token in self.split_tokens(frames):
            if token.startswith("p"):
                t[self.encode_map[token]] = 1
        return t

    def decode(self, x: torch.Tensor, clean: bool = False) -> list | tuple:
        decoded = []
        for token in x.tolist():
            if token in self.decode_map:
                decoded.append(self.decode_map[token])
            else:
                decoded.append(self.unk_token)
        if clean:
            return self.clean(decoded)
        return decoded

    def decode_batch(self, x: Iterable[torch.Tensor], clean: bool = False) -> list | tuple:
        return [self.decode(y, clean) for y in x]

    def clean(self, x: list[str]) -> tuple:
        """Remove special tokens from the decoded text"""
        angle, grade = None, None
        frames = ""
        start = x.index(self.bos_token) if self.bos_token in x else 0
        end = x.index(self.eos_token) if self.eos_token in x else len(x)
        x = x[start + 1 : end]
        for i in x:
            if i.startswith("a"):
                angle = i
            elif i.startswith("f"):
                grade = i
            elif i.startswith("p") or i.startswith("r"):
                frames += i
        return frames, angle, grade

    def save(self, path: str):
        torch.save(self, path)

    @staticmethod
    def load(path: str) -> "Tokenizer":
        return torch.load(path)

    def pad(self, x: torch.Tensor, size: int, where: Literal["left", "right"] = "left"):
        return pad_to(x, size, self.encode_map[self.pad_token], where=where)

    def __repr__(self):
        return f"Tokenizer, tokens:{len(self.encode_map)}, hold:{len(self.hold_tokens)}, angle:{len(self.angle_tokens)}, grade:{len(self.grade_tokens)}"
