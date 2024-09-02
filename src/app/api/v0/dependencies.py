"""
Defines dependencies from service layer.
"""

from src.app.repository.generations import GenerationsRepository
from src.app.service.kilter_gpt import KilterService


def get_kilter_service() -> KilterService:
    return KilterService(GenerationsRepository)
