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
    return " ".join(["p" + x for x in holds])


class Tokenizer:
    def __init__(self, data: Iterable[str]):
        self.data = data
        self.encode_map = self._get_token_map()
        self.decode_map = {v: k for k, v in self.encode_map.items()}

    @staticmethod
    def split_tokens(frames: str) -> list[str]:
        """If not whitespace separated, split the frames into tokens."""
        res = []
        for pair in frames.split("p")[1:]:
            hold, color = pair.split("r")
            res += [f"p{hold}", f"r{color}"]
        return res

    def encode(self, frames: str) -> torch.Tensor:
        """Assumes whitespace separated"""
        split = ["[BOS]"] + frames.split() + ["[EOS]"]
        return torch.tensor([self.encode_map[x] for x in split], dtype=torch.long)

    def encode_batch(self, frames: list[str]) -> list[torch.Tensor]:
        return [self.encode(x) for x in frames]

    def decode(self, x: torch.Tensor) -> str:
        decoded = []
        for token in x.tolist():
            if token in self.decode_map:
                decoded.append(self.decode_map[token])
            else:
                decoded.append("[UNK]")
        return " ".join(decoded)

    def decode_batch(self, x: Iterable[torch.Tensor]) -> list[str]:
        return [self.decode(y) for y in x]

    def _get_token_map(self) -> dict[str, int]:
        """Assumes whitespace separated frames."""
        tokens = ["[BOS]", "[EOS]", "[MASK]", "[UNK]", "[PAD]", "r12", "r13", "r14", "r15"]
        for frame in self.data:
            for token in frame.split():
                if token not in tokens:
                    tokens.append(token)
        encode_map = {token: idx for idx, token in enumerate(tokens)}
        return encode_map


class EncoderDecoder:
    """Converts frames to tensors and back.
    If given tensor - returns string and angle.
    If given string and angle - returns (5,48,48) tensor.
    """

    def __init__(self):
        holds = pd.read_csv("data/raw/holds.csv", index_col=0)
        image_coords = pd.read_csv("figs/image_coords.csv", index_col=0)
        self.coord_to_id = self._create_coord_to_id(holds)
        self.id_to_coord = self._create_id_to_coord(holds)
        self.image_coords = self._create_image_coords(image_coords)

    def _create_coord_to_id(self, holds: pd.DataFrame):
        hold_lookup_matrix = np.zeros((48, 48), dtype=int)
        for i in range(48):
            for j in range(48):
                hold = holds[(holds["x"] == (i * 4 + 4)) & (holds["y"] == (j * 4 + 4))]
                if not hold.empty:
                    hold_lookup_matrix[i, j] = int(hold.index[0])
        return hold_lookup_matrix

    def _create_id_to_coord(self, holds):
        id_to_coord = holds[["x", "y"]]
        id_to_coord = (id_to_coord - 4) // 4
        return id_to_coord.transpose().to_dict(orient="list")

    def _create_image_coords(self, image_coords: pd.DataFrame):
        return {name: (row["x"], row["y"]) for name, row in image_coords.iterrows()}

    def str_to_tensor(self, frames: str, angle: float) -> torch.Tensor:
        angle_matrix = torch.ones((1, 48, 48), dtype=torch.float32) * (angle / 70)
        matrix = torch.zeros((4, 48, 48), dtype=torch.float32)
        for frame in frames.split("p")[1:]:
            hold_id, color = frame.split("r")
            hold_id, color = int(hold_id), int(color) - 12
            coords = self.id_to_coord[hold_id]
            matrix[color, coords[0], coords[1]] = 1
        return torch.cat((matrix, angle_matrix), dim=0)

    def tensor_to_str(self, matrix: torch.Tensor) -> str:
        angle = ((matrix[-1].mean() * 70 / 5).round() * 5).long().item()
        matrix = matrix[:-1, :, :].round().long()
        frames = []
        counter = [0, 0, 0, 0]
        for color, x, y in zip(*torch.where(matrix)):
            counter[color] += 1
            color, x, y = color.item(), x.item(), y.item()
            # too many start/end holds
            if counter[color] > 2 and color in [0, 2]:
                continue
            hold_id = self.coord_to_id[x, y]
            # wrong hold position
            if hold_id == 0:
                continue
            role = color + 12
            frames.append((hold_id, role))
        sorted_frames = sorted(frames, key=lambda x: x[0])
        return ("".join([f"p{hold_id}r{role}" for hold_id, role in sorted_frames]), angle)

    def plot_climb(self, frames: str):
        assert isinstance(frames, str), f"Input must be frames! Got {type(frames)}"
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
        except Exception as e:
            pass
        return image

    def __call__(self, *args):
        if len(args) == 1:
            return self.tensor_to_str(*args)
        elif len(args) == 2:
            return self.str_to_tensor(*args)
        else:
            raise ValueError(f"Only 2 input args allowed! You provided {len(args)}")


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
