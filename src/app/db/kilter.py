import datetime
from dataclasses import dataclass, field


class TableNames:
    climbs_table = "boards_kilter__climb_stats"
    climb_stats_table = "boards_kilter__climbs"
    climb_feedback_table = "boards_kilter__climb_feedback"
    generation_metadata_table = "boards_kilter__climb_generation_metadata"
    generated_climb_feedback_table = "boards_kilter__climb_generation_metadata"


class DictMixin:
    """Mixin class to add 'dict()' method for dataclasses."""
    def dict(self):
        return {k: v for k, v in self.__dict__.items()}


@dataclass
class KilterClimbStats(DictMixin):
    climb_id: int
    angle: int
    difficulty: int
    ascension_count: int
    quality: float
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)


class KilterClimbs(DictMixin):
    id: int
    layout_id: int
    name: str
    description: str
    frames: str
    generated: bool = True
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)


class GenerationMetadata(DictMixin):
    climb_id: int
    frames: str
    temperature: float
    top_p_sampling: float
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)


class GeneratedClimbFeedback(DictMixin):
    climb_id: int
    user_id: int
    type: str
    created_at: datetime.datetime = field(default_factory=datetime.datetime.now)
