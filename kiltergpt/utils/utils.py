import math
from collections import Counter

from ..data.tokenizer import Tokenizer


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


class KilterPolice:
    """Punishes bad climbs."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        n_start_holds: tuple[int, int] = (1, 2),
        n_finish_holds: tuple[int, int] = (1, 2),
        n_total_holds: tuple[int, int] = (2, math.inf),
    ):
        self.allowed_colors = set([int(x[1:]) for x in tokenizer.color_tokens()])
        self.allowed_holds = set([int(x[1:]) for x in tokenizer.hold_tokens()])
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
