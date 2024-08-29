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
    climb: str
    feedback: Optional[str]
