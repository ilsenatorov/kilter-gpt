from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Literal


class ClimbGrade(str, Enum):
    _5a = "5a"
    _5a_plus = "5a+"
    _5b = "5b"
    _5b_plus = "5b+"
    _5c = "5c"
    _5c_plus = "5c+"
    _6a = "6a"
    _6a_plus = "6a+"
    _6b = "6b"
    _6b_plus = "6b+"
    _6c = "6c"
    _6c_plus = "6c+"
    _7a = "7a"
    _7a_plus = "7a+"
    _7b = "7b"
    _7b_plus = "7b+"
    _7c = "7c"
    _7c_plus = "7c+"
    _8a = "8a"
    _8a_plus = "8a+"
    _8b = "8b"
    _8b_plus = "8b+"
    # TODO: do we want to generate harder climbs or vice versa limit with softer grades
    # due to generation quality?


class GenerationParams(BaseModel):
    frames: str = Field(description="")
    angle: int = Field(description="Angle of current user's Kilterboard setup", ge=0, le=70)
    grade: ClimbGrade = Field(description="Grade that supposed to be generated")
    temperature: float = Field(description="Responds for randomness of routes generation", default=0.8, ge=0, le=1)
    p: float = Field(description="top-p sampling (nucleus)", default=1.0)


class Climb(BaseModel):
    holds: str = Field(description="Set of generated holds")
    id: Optional[int] = Field(description="Id that should be used to send feedback with", default=None)


class Feedback(BaseModel):
    feedback: bool | None = Field(
        description="True/False for like/dislike, None if not provided",
        default=None
    )
    climb_id: int = Field(description="Id for the climb you want to send feedback for")


class Generation(BaseModel):
    params: GenerationParams
    climb: Climb
    feedback: Optional[Feedback] = None
