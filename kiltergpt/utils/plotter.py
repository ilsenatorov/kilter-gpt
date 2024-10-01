import warnings

import cv2
import matplotlib
import pandas as pd
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

    def __init__(self, colormap: str = "viridis", verbose_cutoff: float = 0.1):
        self.image_coords = self._create_image_coords(pd.read_csv("figs/image_coords.csv", index_col=0))
        self.colormap = matplotlib.colormaps[colormap]
        self.verbose_cutoff = verbose_cutoff

    def _create_image_coords(self, image_coords: pd.DataFrame):
        return {name: (row["img_x"], row["img_y"]) for name, row in image_coords.iterrows()}

    def plot_climb(
        self,
        frames: str,
        return_fig: bool = False,
        highlight: str | None = None,
        probs: dict | None = None,
    ):
        assert all(x in "0123456789pr" for x in frames), "Frames should only contain p, r and digits"
        frames = frames.replace(" ", "")  # here the input takes no whitespace
        board_path = "figs/full_board_commercial.png"
        image = cv2.imread(board_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
        for hold in frames.split("p")[1:]:
            try:
                hold_id, hold_type = hold.split("r")
            except ValueError:
                warnings.warn(f"Can't split hold/color pair in {frames}", stacklevel=2)
                continue
            if int(hold_id) not in self.image_coords:
                warnings.warn(f"Hold {hold_id} not in image coordinates", stacklevel=2)
                continue
            radius = 30
            thickness = 3
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
                thickness = 2
                image = cv2.circle(image, self.image_coords[int(hold_id)], radius, self.highlight_color, thickness)
        if probs is not None:
            for hold, prob in probs.items():
                if not hold.startswith("p"):
                    continue
                hold = int(hold[1:])
                if hold not in self.image_coords:
                    continue
                radius = 24
                thickness = 2
                color = self.colormap(prob)
                color = tuple(int(channel * 255) for channel in color[:3])
                image = cv2.circle(image, self.image_coords[int(hold)], radius, color, thickness)
                coords = self.image_coords[int(hold)]
                coords = (coords[0] - 10, coords[1])
                if prob > self.verbose_cutoff:
                    image = cv2.putText(
                        image,
                        f"{int(prob*100)}%",
                        coords,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                        bottomLeftOrigin=False,
                    )
        if return_fig:
            return plt.imshow(image)
        return image

    def __call__(self, frames: str, return_fig: bool = False, highlight: str | None = None):
        return self.plot_climb(frames, return_fig, highlight)
