"""
Defines dependencies from service layer.
"""
from functools import lru_cache

from src.app.repository.kilter import KilterRepository
from src.app.service.kilter_gpt import KilterService


@lru_cache(1)
def get_kilter_service() -> KilterService:
    return KilterService(KilterRepository)
