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
    }


def matching_holds(orig, ref, dim: int = 1):
    """Fraction of matching holds."""
    matching = (orig * ref).sum(dim=dim)
    return matching / orig.sum(dim=dim)
