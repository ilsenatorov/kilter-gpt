from pydantic import BaseModel
from typing import Optional


class GenerationParams(BaseModel):
    frames: str
    angle: int
    grade: str
    temperature: float = 0.2
    p: float = 1.0  # TODO: come up with more representative name?


class Generation(BaseModel):
    params: GenerationParams
    climb: str  # TODO: think about extracting it to an independent model in case if we will decide to make it all in one service for all kilter climbs?
                # it will mean, that current solution to store all data in the single table is not the most genious idea.
    feedback: Optional[str]
