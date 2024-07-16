from typing import Iterable, Literal

import cv2
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

colors = torch.tensor(
    [
        [0, 255, 0],  # Green
        [0, 0, 255],  # Blue
        [128, 0, 128],  # Purple
        [255, 165, 0],  # Orange
    ],
    dtype=torch.uint8,
)


def shuffle_holds(climb: str) -> str:
    """Shuffle the holds in a climb"""
    holds = climb.split("p")[1:]
    np.random.shuffle(holds)
    return "".join(["p" + x.strip() for x in holds])


class Tokenizer:
    eos_token = "[EOS]"
    bos_token = "[BOS]"
    pad_token = "[PAD]"
    unk_token = "[UNK]"
    mask_token = "[MASK]"

    def __init__(self, encode_map: dict[str, int], angle: bool = False, grade: bool = False):
        self.encode_map = encode_map
        self.decode_map = {v: k for k, v in self.encode_map.items()}
        self.angle = angle
        self.grade = grade
        self._set_special_token_ids()

    @property
    def angle_tokens(self):
        return [x for x in self.encode_map if x.startswith("a")]

    @property
    def grade_tokens(self):
        return [x for x in self.encode_map if x.startswith("f")]

    def _set_special_token_ids(self):
        self.pad_token_id = self.encode_map[self.pad_token]
        self.bos_token_id = self.encode_map[self.bos_token]
        self.eos_token_id = self.encode_map[self.eos_token]
        self.unk_token_id = self.encode_map[self.unk_token]
        self.mask_token_id = self.encode_map[self.mask_token]

    @staticmethod
    def from_df(df: pd.DataFrame, angle: bool = False, grade: bool = False) -> "Tokenizer":
        tokens = [
            Tokenizer.bos_token,
            Tokenizer.eos_token,
            Tokenizer.pad_token,
            Tokenizer.mask_token,
            Tokenizer.unk_token,
            "r12",
            "r13",
            "r14",
            "r15",
        ]
        ## Add all unique tokens from the frames
        for frame in df["frames"]:
            for token in Tokenizer.split_tokens(frame):
                if token not in tokens:
                    tokens.append(token)
        # Add angle and difficulty tokens
        if angle:
            for i in df["angle"].unique():
                tokens.append(f"a{i}")
        if grade:
            for i in df["font_grade"].unique():
                tokens.append(f"f{i}")
        encode_map = {token: idx for idx, token in enumerate(tokens)}
        return Tokenizer(encode_map, angle, grade)

    @staticmethod
    def split_tokens(frames: str) -> list[str]:
        """If not whitespace separated, split the frames into tokens."""
        assert " " not in frames, "No whitespace allowed in frames"
        res = []
        for pair in frames.split("p")[1:]:
            hold, color = pair.split("r")
            res += [f"p{hold}", f"r{color}"]
        return res

    def encode(
        self,
        frames: str,
        angle: int = None,
        grade: str = None,
        bos: bool = True,
        eos: bool = True,
        pad: int = 0,
    ) -> torch.Tensor:
        tokens = []
        if bos:
            tokens.append(self.bos_token)
        if self.angle and angle:
            tokens.append(f"a{angle}")
        if self.grade and grade:
            tokens.append(f"f{grade}")
        tokens += self.split_tokens(frames)
        if eos:
            tokens.append(self.eos_token)
        t = torch.tensor([self.encode_map[x] for x in tokens], dtype=torch.long)
        if pad:
            t = self.pad(t, pad)
        return t

    def encode_batch(self, frames: list[str]) -> list[torch.Tensor]:
        return [self.encode(x) for x in frames]

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
        return f"Tokenizer(angle={self.angle}, grade={self.grade})"


class Plotter:
    """Plots the selected holds onto the empty kilterboard. Requires df from `figs/` folder."""

    def __init__(self):
        self.image_coords = self._create_image_coords(pd.read_csv("figs/image_coords.csv", index_col=0))

    def _create_image_coords(self, image_coords: pd.DataFrame):
        return {name: (row["img_x"], row["img_y"]) for name, row in image_coords.iterrows()}

    def plot_climb(self, frames: str, return_fig: bool = False):
        frames = frames.replace(" ", "")  # here the input takes no whitespace
        board_path = "figs/full_board_commercial.png"
        image = cv2.imread(board_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        try:
            for hold in frames.split("p")[1:]:
                hold_id, hold_type = hold.split("r")
                if int(hold_id) not in self.image_coords:
                    continue
                radius = 30
                thickness = 2
                if hold_type == str(12):
                    color = (0, 255, 0)  # start
                if hold_type == str(13):  # hands
                    color = (0, 200, 255)
                if hold_type == str(14):  # end
                    color = (255, 0, 255)
                if hold_type == str(15):  # feet
                    color = (255, 165, 0)
                image = cv2.circle(image, self.image_coords[int(hold_id)], radius, color, thickness)
        except Exception as e:  # FIXME
            print(e)
        if return_fig:
            return plt.imshow(image)
        return image


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
    return torch.nn.functional.pad(tensor, (left_pad, right_pad), value=pad_value)


def str_to_bool(value: str) -> bool:
    """Command line inputs that are bools."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ValueError("Boolean value expected.")
