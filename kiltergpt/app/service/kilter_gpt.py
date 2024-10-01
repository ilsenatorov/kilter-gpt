"""
Kilter service.
Implements KilterGPT inference.
Work with DB is incapsulated via KilterRepository (feedback&generations).
"""

import logging
from typing import Type

import torch

from kiltergpt.app.config import settings
from kiltergpt.app.models.generation import Climb, Feedback, GenerationParams
from kiltergpt.app.repository.kilter import KilterRepository
from kiltergpt.models import GPTModel
from kiltergpt.utils.hasher import Hasher


class KilterService:
    def __init__(self, data_repository: Type[KilterRepository]) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Initialized {self.__class__.__name__}")

        self.data_repository: KilterRepository = data_repository()
        self.kilter_gpt = self.load_model()
        self.hasher = Hasher.from_json()

    def load_model(self):
        model = GPTModel.load_from_wandb(settings.WANDB_MODEL_CHECKPOINT).to("cpu")
        model.eval()
        return model

    def generate_route(self, generation_params: GenerationParams) -> Climb:
        """Generates route and saves it to repository."""
        self.logger.info("Starting route generation...")

        # TODO: maybe extract torch context manager into GPTModel.generate_from_string()
        # since it's an extra dependency for the service layer?
        with torch.no_grad():
            holds = self.kilter_gpt.generate_from_string(
                frames=generation_params.frames,
                angle=generation_params.angle,
                grade=generation_params.grade,
                temperature=generation_params.temperature,
                p=generation_params.p,
            )
        entropy = self.kilter_gpt.get_entropy()
        climb_name = self.hasher.encode(holds)
        climb = Climb(holds=holds, name=climb_name, entropy=entropy)
        # idea is to return generation even if we failed to save it
        try:
            climb.id = self.data_repository.save_generation(
                holds=holds, climb_name=climb_name, generation_params=generation_params
            )
        except Exception as e:
            self.logger.error(f"Failed to save generated climb: {e}")

        return climb

    def save_feedback(self, feedback: Feedback) -> None:
        """Updates feedback by climbs id."""
        self.data_repository.save_feedback(feedback)
