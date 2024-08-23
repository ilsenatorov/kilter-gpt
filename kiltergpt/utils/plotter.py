import cv2
import pandas as pd
import torch
from matplotlib import pyplot as plt

# [0, 255, 0],  # Green
# [0, 0, 255],  # Blue
# [128, 0, 128],  # Purple
# [255, 165, 0],  # Orange
# [230, 230, 230],  # White


class Plotter:
    """Plots the selected holds onto the empty kilterboard. Requires df from `figs/` folder."""

    start_color = (0, 255, 0)
    hand_color = (0, 200, 255)
    finish_color = (255, 0, 255)
    foot_color = (255, 165, 0)
    highlight_color = (255, 255, 255)

    def __init__(self):
        self.image_coords = self._create_image_coords(pd.read_csv("figs/image_coords.csv", index_col=0))

    def _create_image_coords(self, image_coords: pd.DataFrame):
        return {name: (row["img_x"], row["img_y"]) for name, row in image_coords.iterrows()}

    def plot_climb(self, frames: str, return_fig: bool = False, highlight: str = None):
        assert all(x in "0123456789pr" for x in frames), "Frames should only contain p, r and digits"
        frames = frames.replace(" ", "")  # here the input takes no whitespace
        board_path = "figs/full_board_commercial.png"
        image = cv2.imread(board_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        for hold in frames.split("p")[1:]:
            hold_id, hold_type = hold.split("r")
            if int(hold_id) not in self.image_coords:
                continue
            radius = 30
            thickness = 2
            match hold_type:
                case "12":
                    color = self.start_color
                case "13":
                    color = self.hand_color
                case "14":
                    color = self.finish_color
                case "15":
                    color = self.foot_color
                case _:
                    raise ValueError(f"Unknown hold color {color}")
            image = cv2.circle(image, self.image_coords[int(hold_id)], radius, color, thickness)
        if highlight is not None:
            for hold_id in highlight.split("p")[1:]:
                if int(hold_id) not in self.image_coords:
                    continue
                radius = 24
                thickness = 3
                image = cv2.circle(image, self.image_coords[int(hold_id)], radius, self.highlight_color, thickness)
        if return_fig:
            return plt.imshow(image)
        return image

    def __call__(self, frames: str, return_fig: bool = False, highlight: str = None):
        return self.plot_climb(frames, return_fig, highlight)
