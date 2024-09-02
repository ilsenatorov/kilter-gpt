"""
Kilter service.
Implements KilterGPT inference.
Work with DB is incapsulated via KilterRepository (feedback&generations).
"""

import logging
from typing import Type
from kiltergpt.models.gpt import GPTModel
from src.app.config import settings
from src.app.models.generation import Feedback, GenerationParams, Generation, Climb
from src.app.repository.generations import GenerationsRepository


class KilterService:
    def __init__(self, data_repository: Type[GenerationsRepository]) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__}")

        self.data_repository: GenerationsRepository = data_repository()
        self.kilter_gpt = GPTModel.load_from_wandb(settings.WANDB_MODEL_NAME).to("cpu")  # TODO: maybe make device dependent on CUDA existence?

    def generate_route(self, generation_params: GenerationParams) -> Climb:
        """Generates route and saves it to repository."""

        holds = self.kilter_gpt.generate_from_string(
            frames=generation_params.frames,
            angle=generation_params.angle,
            grade=generation_params.grade,
            temperature=generation_params.temperature,
            p=generation_params.p
        )

        climb = Climb(holds=holds)

        climb.id = self.data_repository.save_generation(Generation(
            params=generation_params,
            climb=climb
        ))

        return climb

    def update_feedback(self, feedback: Feedback) -> None:
        """Updates feedback by climbs id."""
        self.data_repository.update_feedback(feedback)
