from functools import cache
from itertools import pairwise
from random import Random

import pandas as pd
import torch
import torch.nn as nn

from ..data.datasets import climb_signature
from ..data.tokenizer import IMAGE_COORDS
from .utils import KilterPolice

ROLES = ("12", "13", "14", "15")  # start, hand, finish, foot


@cache
def board_positions() -> dict[str, tuple[float, float]]:
    """Placement id -> (x, y) on the board, both in [0, 1] from the bottom left"""
    coords = pd.read_csv(IMAGE_COORDS, index_col=0)
    x, y = coords.img_x / coords.img_x.max(), 1 - coords.img_y / coords.img_y.max()
    return {str(placement): (x[placement], y[placement]) for placement in coords.index}


def _holds(frames: str) -> list[tuple[str, str]]:
    return [tuple(hold.split("r")) for hold in frames.split("p")[1:]]


def max_hand_gap(frames: str) -> float:
    """Largest vertical gap (fraction of board height) between consecutive hand holds, feet excluded"""
    ys = sorted(board_positions()[hold][1] for hold, role in _holds(frames) if role != "15")
    return max((high - low for low, high in pairwise(ys)), default=0.0)


def climb_features(climbs: list[str]) -> torch.Tensor:
    """(hold, role) multi-hot plus role counts and the hand gap, for the two-sample test"""
    index = {placement: i for i, placement in enumerate(board_positions())}
    features = torch.zeros(len(climbs), len(index) * len(ROLES) + len(ROLES) + 1)
    for i, frames in enumerate(climbs):
        for hold, role in _holds(frames):
            features[i, index[hold] * len(ROLES) + ROLES.index(role)] = 1
            features[i, -1 - len(ROLES) + ROLES.index(role)] += 0.1
        features[i, -1] = max_hand_gap(frames)
    return features


def c2st_auc(real: list[str], generated: list[str], folds: int = 5, seed: int = 0) -> float:
    """Classifier two-sample test: cross-validated ROC AUC of an MLP telling generated from real climbs.
    0.5 means the classifier can't tell them apart; 1.0 means every generated climb gives itself away.
    real[i] and generated[i] are paired (generated from real[i]'s prompt and conditions). Climbs repeat at several
    angles, so only the first pair of each real climb is kept: a copy on both sides of a fold would leak."""
    if len(real) != len(generated):
        raise ValueError("c2st_auc needs one generated climb per real climb")
    first = {climb_signature(frames): i for i, frames in reversed(list(enumerate(real)))}
    real, generated = [real[i] for i in sorted(first.values())], [generated[i] for i in sorted(first.values())]
    n = len(real)
    pairs = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    with torch.random.fork_rng(), torch.inference_mode(False), torch.enable_grad():  # Lightning tests without grad
        torch.manual_seed(seed)
        scores = torch.empty(2 * n)
        x = climb_features(real + generated)
        y = torch.cat((torch.zeros(n), torch.ones(n)))
        for fold in pairs.chunk(folds):  # a pair shares its prompt holds, so both sides stay in one fold
            fold = torch.cat((fold, fold + n))
            train = torch.arange(2 * n)[~torch.isin(torch.arange(2 * n), fold)]
            net = nn.Sequential(nn.Linear(x.size(1), 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
            optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
            for _ in range(200):  # full batch
                loss = nn.functional.binary_cross_entropy_with_logits(net(x[train]).squeeze(1), y[train])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                scores[fold] = net.eval()(x[fold]).squeeze(1)
    generated_scores, real_scores = scores[n:, None], scores[None, :n]  # Mann-Whitney U, ties count half
    return ((generated_scores > real_scores).float() + 0.5 * (generated_scores == real_scores).float()).mean().item()


def generation_metrics(
    generated: list[str], reference: set[str], police: KilterPolice, real: list[str] | None = None
) -> dict[str, float]:
    """real: held-out climbs to run the two-sample test against"""
    if not generated or not reference:
        raise ValueError("Generation metrics require generated and reference climbs")
    sampled = [
        set(signature.split("|")) for signature in Random(42).sample(sorted(reference), min(1024, len(reference)))
    ]
    signatures = [climb_signature(frames) for frames in generated]
    distances = []
    for signature in signatures:
        holds = set(signature.split("|")) if signature else set()
        closest = max((len(holds & other) / len(holds | other) for other in sampled), default=0)
        distances.append(1 - closest)
    return {
        "valid_rate": sum(police.check(frames) for frames in generated) / len(generated),
        "novel_rate": sum(signature not in reference for signature in signatures) / len(generated),
        "nearest_sampled_distance": sum(distances) / len(distances),
        **structure_metrics(generated),
        **({"c2st_auc": c2st_auc(real, generated)} if real else {}),
    }


def structure_metrics(climbs: list[str]) -> dict[str, float]:
    """How climbs are built, to compare generated with real: sampling below temperature 1 drops foot holds first"""
    if not climbs:
        raise ValueError("Structure metrics require climbs")
    holds = [frames.count("p") for frames in climbs]
    feet = [frames.count("r15") for frames in climbs]
    return {
        "mean_holds": sum(holds) / len(climbs),
        "mean_feet": sum(feet) / len(climbs),
        "no_feet_rate": sum(n == 0 for n in feet) / len(climbs),
        "max_hand_gap": sum(max_hand_gap(frames) for frames in climbs) / len(climbs),
    }
