from random import Random

from ..data.datasets import climb_signature
from .utils import KilterPolice


def generation_metrics(generated: list[str], reference: set[str], police: KilterPolice) -> dict[str, float]:
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
    }
