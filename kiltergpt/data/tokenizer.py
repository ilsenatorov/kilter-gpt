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


class Tokenizer:
    def __init__(self):
        self.encode_map: dict[str, int] = dict()
        for i, token in enumerate(self.special_tokens() + self.color_tokens() + self.hold_tokens()):
            self.encode_map[token] = i
        self.decode_map = {v: k for k, v in self.encode_map.items()}
        self._set_special_tokens()

    @staticmethod
    def hold_tokens():
        res = []
        for i in range(1073, 1600):
            # Excludes the 12x14 holds
            if i < 1396 or i > 1446:
                res.append(f"p{i}")
        return res

    @staticmethod
    def start_token() -> str:
        return "r12"

    @staticmethod
    def handhold_token() -> str:
        return "r13"

    @staticmethod
    def finish_token() -> str:
        return "r14"

    @staticmethod
    def foothold_token() -> str:
        return "r15"

    @staticmethod
    def color_tokens():
        return [
            Tokenizer.start_token(),
            Tokenizer.handhold_token(),
            Tokenizer.finish_token(),
            Tokenizer.foothold_token(),
        ]

    @staticmethod
    def grade_tokens():
        return [
            "1a",
            "1b",
            "1c",
            "2a",
            "2b",
            "2c",
            "3a",
            "3b",
            "3c",
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
            "9a",
            "9a+",
            "9b",
            "9b+",
            "9c",
            "9c+",
        ]

    @property
    def start_token_id(self) -> int:
        return self.encode_map[self.start_token()]

    @property
    def handhold_token_id(self) -> int:
        return self.encode_map[self.handhold_token()]

    @property
    def finish_token_id(self) -> int:
        return self.encode_map[self.finish_token()]

    @property
    def foothold_token_id(self) -> int:
        return self.encode_map[self.foothold_token()]

    @staticmethod
    def special_tokens():
        return ["[PAD]", "[BOS]", "[EOS]", "[UNK]", "[MASK]"]

    @property
    def hold_token_ids(self):
        return torch.tensor([self.encode_map[x] for x in self.hold_tokens()])

    @property
    def color_token_ids(self):
        return torch.tensor([self.encode_map[x] for x in self.color_tokens()])

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
        angle: int | None = None,
        grade: str | float | None | np.int64 = None,
        *,
        shuffle: bool = False,
        bos: bool = True,
        eos: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        assert all(x in "0123456789pr" for x in frames), "Frames should only contain p, r and digits"
        tokens = []
        if bos:
            tokens.append(self.bos_token)
        if shuffle:
            frames = shuffle_holds(frames)
        tokens.extend(self.split_tokens(frames))
        if eos:
            tokens.append(self.eos_token)
        t = torch.tensor([self.encode_map.get(x, self.unk_token_id) for x in tokens], dtype=torch.long)
        if angle is not None:
            angle = torch.tensor(angle / 70.0, dtype=torch.float32)
        if grade is not None:
            if isinstance(grade, str):
                grade = torch.tensor(
                    ((self.grade_tokens().index(grade)) + 1) / len(self.grade_tokens()), dtype=torch.float32
                )
            else:
                grade = torch.tensor(grade / len(self.grade_tokens()), dtype=torch.float32)
        return t, angle, grade

    def onehot(self, frames: str | torch.Tensor) -> torch.Tensor:
        """Save presence/absence of each hold in a one-hot tensor"""
        if isinstance(frames, str):
            return self._onehot_from_string(frames)
        return self._onehot_from_tensor(frames)

    def _onehot_from_string(self, frames: str) -> torch.Tensor:
        t = torch.zeros(len(self.encode_map), dtype=torch.bool)
        for token in self.split_tokens(frames):
            if token.startswith("p"):
                t[self.encode_map[token]] = True
        return t

    def _onehot_from_tensor(self, encoded_frames: torch.Tensor) -> torch.Tensor:
        t = torch.zeros(len(self.encode_map), dtype=torch.bool)
        encoded_frames = encoded_frames[torch.isin(encoded_frames, self.hold_token_ids)]
        t[encoded_frames] = True
        return t

    def decode(self, x: torch.Tensor | list, clean: bool = False) -> list[str] | str:
        decoded = []
        if isinstance(x, torch.Tensor):
            x = x.tolist()
        for token in x:
            if token in self.decode_map:
                decoded.append(self.decode_map[token])
            else:
                decoded.append(self.unk_token)
        if clean:
            return self.clean(decoded)
        return decoded

    def clean(self, x: list[str]):
        """Remove special tokens from the decoded text"""
        frames = ""
        start = x.index(self.bos_token) if self.bos_token in x else 0
        end = x.index(self.eos_token) if self.eos_token in x else len(x)
        x = x[start + 1 : end]
        for i in x:
            if i.startswith("p") or i.startswith("r"):
                frames += i
        return frames

    def __repr__(self):
        return f"Tokenizer, tokens:{len(self.encode_map)}, hold:{len(self.hold_tokens())}"
