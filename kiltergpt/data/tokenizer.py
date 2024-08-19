import json
from typing import Literal

import numpy as np
import pandas as pd
import torch


def shuffle_holds(climb: str) -> str:
    """Shuffle the holds in a climb"""
    holds = climb.split("p")[1:]
    np.random.shuffle(holds)
    return "".join(["p" + x.strip() for x in holds])


def sort_holds(climb: str) -> str:
    """Sort the holds in a climb"""
    holds = climb.split("p")[1:]
    holds.sort()
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
    def __init__(self):
        self.encode_map: dict[str, int] = dict()
        for i, token in enumerate(
            self.special_tokens()
            + self.color_tokens()
            + self.hold_tokens()
            + self.angle_tokens()
            + self.grade_tokens()
        ):
            self.encode_map[token] = i
        self.decode_map = {v: k for k, v in self.encode_map.items()}
        self._set_special_tokens()

    @staticmethod
    def angle_tokens():
        return [f"a{i}" for i in range(0, 95, 5)]

    @staticmethod
    def hold_tokens():
        return [f"p{i}" for i in range(1073, 1600)]

    @staticmethod
    def color_tokens():
        return ["r12", "r13", "r14", "r15"]

    @staticmethod
    def grade_tokens():
        return [
            f"f{i}"
            for i in [
                "4a",
                "4b",
                "4c",
                "5a",
                "5b",
                "5c",
                "6a",
                "6a+",
                "6b",
                "6b+",
                "6c",
                "6c+",
                "7a",
                "7a+",
                "7b",
                "7b+",
                "7c",
                "7c+",
                "8a",
                "8a+",
                "8b",
                "8b+",
                "8c",
                "8c+",
            ]
        ]

    @staticmethod
    def special_tokens():
        return ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[MASK]"]

    @property
    def angle_token_ids(self):
        return torch.tensor([self.encode_map[x] for x in self.angle_tokens()])

    @property
    def hold_token_ids(self):
        return torch.tensor([self.encode_map[x] for x in self.hold_tokens()])

    @property
    def color_token_ids(self):
        return torch.tensor([self.encode_map[x] for x in self.color_tokens()])

    @property
    def grade_token_ids(self):
        return torch.tensor([self.encode_map[x] for x in self.grade_tokens()])

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

    def _set_special_tokens(self):
        self.pad_token = self.special_tokens()[0]
        self.bos_token = self.special_tokens()[1]
        self.eos_token = self.special_tokens()[2]
        self.unk_token = self.special_tokens()[3]
        self.mask_token = self.special_tokens()[4]
        self.pad_token_id = self.encode_map[self.special_tokens()[0]]
        self.bos_token_id = self.encode_map[self.special_tokens()[1]]
        self.eos_token_id = self.encode_map[self.special_tokens()[2]]
        self.unk_token_id = self.encode_map[self.special_tokens()[3]]
        self.mask_token_id = self.encode_map[self.special_tokens()[4]]

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

    @property
    def vocab_size(self):
        return len(self.encode_map)

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
        assert " " not in frames, "Frames should not contain spaces"
        assert all(x in "0123456789pr" for x in frames), "Frames should only contain p, r and digits"
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

    def clean(self, x: list[str]) -> tuple:
        """Remove special tokens from the decoded text"""
        angle, grade = None, None
        frames = ""
        start = x.index(self.bos_token) if self.bos_token in x else 0
        end = x.index(self.eos_token) if self.eos_token in x else len(x)
        x = x[start + 1 : end]
        for i in x:
            if i.startswith("a"):
                angle = int(i[1:])
            elif i.startswith("f"):
                grade = i[1:]
            elif i.startswith("p") or i.startswith("r"):
                frames += i
        return frames, angle, grade

    def pad(self, x: torch.Tensor, size: int, where: Literal["left", "right"] = "left"):
        return pad_to(x, size, self.encode_map[self.pad_token], where=where)

    def __repr__(self):
        return f"Tokenizer, tokens:{len(self.encode_map)}, hold:{len(self.hold_tokens)}, angle:{len(self.angle_tokens)}, grade:{len(self.grade_tokens)}"
