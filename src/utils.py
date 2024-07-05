from typing import Iterable, Literal

import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch

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

    @staticmethod
    def from_df(df: pd.DataFrame, angle: bool = False, grade: bool = False):
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

    def encode(self, frames: str, angle: int = None, grade: str = None) -> torch.Tensor:
        angle, grade = f"a{angle}" or "", f"f{grade}" or ""
        split = [self.bos_token] + [angle] + [grade] + self.split_tokens(frames) + [self.eos_token]
        return torch.tensor([self.encode_map[x] for x in split], dtype=torch.long)

    def encode_batch(self, frames: list[str]) -> list[torch.Tensor]:
        return [self.encode(x) for x in frames]

    def decode(self, x: torch.Tensor) -> list:
        decoded = []
        for token in x.tolist():
            if token in self.decode_map:
                decoded.append(self.decode_map[token])
            else:
                decoded.append(self.unk_token)
        return decoded

    def decode_batch(self, x: Iterable[torch.Tensor]) -> list[str]:
        return [self.decode(y) for y in x]

    def save(self, path: str):
        torch.save(self, path)

    @staticmethod
    def load(path: str):
        return torch.load(path)


class Plotter:
    """Plots the selected holds onto the empty kilterboard. Requires df from `figs/` folder."""

    def __init__(self):
        image_coords = pd.read_csv("figs/image_coords.csv", index_col=0)
        self.image_coords = self._create_image_coords(image_coords)

    def _create_image_coords(self, image_coords: pd.DataFrame):
        return {name: (row["x"], row["y"]) for name, row in image_coords.iterrows()}

    def plot_climb(self, frames: str, return_fig: bool = False):
        # FIXME currently has issues with footholds, check diff with old version
        frames = frames.replace(" ", "")  # here the input takes no whitespace
        board_path = "figs/full_board_commercial.png"
        image = cv2.imread(board_path)
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
            pass
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
