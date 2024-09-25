from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Literal


class ClimbGrade(str, Enum):
    _5a = "5a"
    _5b = "5b"
    _5c = "5c"
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
    _8c = "8c"
    _8c_plus = "8c+"


class GenerationParams(BaseModel):
    frames: str = Field(description="Prompt with the holds, that user want to see in the climb")
    angle: int = Field(description="Angle of current user's Kilterboard setup", ge=0, le=70)
    grade: ClimbGrade = Field(description="Grade that supposed to be generated")
    temperature: float = Field(description="Responds for randomness of routes generation", default=0.8, ge=0.3, le=1)
    p: float = Field(description="top-p sampling (nucleus)", default=1.0)


class Climb(BaseModel):
    holds: str = Field(description="Set of generated holds")
    name: str = Field(description="Generated name for generated climb")
    id: Optional[str] = Field(description="Id that should be used to send feedback with", default=None)  # TODO: not sure about type, maybe UUID?


class FeedbackType(str, Enum):
    like = "like"
    dislike = "dislike"


class Feedback(BaseModel):
    feedback: FeedbackType
    climb_id: str = Field(description="Id (uuid) for the climb you want to send feedback for")
    user_id: int = Field(description="Id of the user, who wants to share feedback")  # TODO: not sure about type, maybe UUID?
