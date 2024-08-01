import math
from collections import Counter

import cv2
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


class WarmupCosineSchedule(torch.optim.lr_scheduler.LambdaLR):
    """Linear warmup and then cosine decay. Coeff is the original lr multiplier."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps: int,
        start_lr_coeff: float,
        end_lr_coeff: float,
        last_epoch=-1,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.start_lr_coeff = start_lr_coeff
        self.end_lr_coeff = end_lr_coeff
        super().__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            progress = step / float(max(1, self.warmup_steps))
            return self.start_lr_coeff + progress * (1 - self.start_lr_coeff)
        elif step < self.total_steps:  # Added this check to cap the decay at total_steps
            progress = (step - self.warmup_steps) / float(max(1, self.total_steps - self.warmup_steps))
            return self.end_lr_coeff + (0.5 * (1 - self.end_lr_coeff) * (1 + math.cos(math.pi * progress)))
        else:
            return self.end_lr_coeff  # Keep the LR at the final value after total_steps


class KilterPolice:
    """Punishes bad climbs."""

    def __init__(
        self,
        allowed_holds: set,
        n_start_holds: tuple[int, int] = (1, 2),
        n_finish_holds: tuple[int, int] = (1, 2),
        n_total_holds: tuple[int, int] = (2, math.inf),
    ):
        self.allowed_holds = allowed_holds
        self.allowed_colors = set([12, 13, 14, 15])
        self.n_start_holds = n_start_holds
        self.n_finish_holds = n_finish_holds
        self.n_total_holds = n_total_holds

    def check(self, frames: str) -> bool:
        """Check if the climb is valid."""
        colors = []
        for frame in frames.split("p")[1:]:  # split by holds
            hold, color = frame.split("r")  # split into hold id and color
            if int(hold) not in self.allowed_holds:
                return False
            if int(color) not in self.allowed_colors:
                return False
            colors.append(int(color))
        if len(colors) < self.n_total_holds[0] or len(colors) > self.n_total_holds[1]:
            return False
        counter = Counter(colors)
        if counter[12] < self.n_start_holds[0] or counter[12] > self.n_start_holds[1]:
            return False
        if counter[14] < self.n_finish_holds[0] or counter[14] > self.n_finish_holds[1]:
            return False
        return True
