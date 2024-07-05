from typing import Iterable, Literal

import cv2
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
    def __init__(self, data: Iterable[str]):
        self.eos_token = "[EOS]"
        self.bos_token = "[BOS]"
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.mask_token = "[MASK]"
        self.data = data
        self.pad_to = pad_to
        self.encode_map = self._get_token_map()
        self.decode_map = {v: k for k, v in self.encode_map.items()}

    @staticmethod
    def split_tokens(frames: str) -> list[str]:
        """If not whitespace separated, split the frames into tokens."""
        assert " " not in frames, "No whitespace allowed in frames"
        res = []
        for pair in frames.split("p")[1:]:
            hold, color = pair.split("r")
            res += [f"p{hold}", f"r{color}"]
        return res

    def encode(self, frames: str) -> torch.Tensor:
        split = [self.bos_token] + self.split_tokens(frames) + [self.eos_token]
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

    def _get_token_map(self) -> dict[str, int]:
        tokens = [
            self.bos_token,
            self.eos_token,
            self.pad_token,
            self.mask_token,
            self.unk_token,
            "r12",
            "r13",
            "r14",
            "r15",
        ]
        for frame in self.data:
            for token in self.split_tokens(frame):
                if token not in tokens:
                    tokens.append(token)
        encode_map = {token: idx for idx, token in enumerate(tokens)}
        return encode_map


class Plotter:
    """Plots the selected holds onto the empty kilterboard. Requires data from `figs/` folder."""

    def __init__(self):
        image_coords = pd.read_csv("figs/image_coords.csv", index_col=0)
        self.image_coords = self._create_image_coords(image_coords)

    def _create_image_coords(self, image_coords: pd.DataFrame):
        return {name: (row["x"], row["y"]) for name, row in image_coords.iterrows()}

    def plot_climb(self, frames: str) -> np.ndarray:
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
