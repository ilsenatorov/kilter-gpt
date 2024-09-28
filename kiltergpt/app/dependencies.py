"""
Defines dependencies from service layer.
"""
from functools import lru_cache

from kiltergpt.app.repository.kilter import KilterRepository
from kiltergpt.app.service.kilter_gpt import KilterService


@lru_cache(1)
def get_kilter_service() -> KilterService:
    return KilterService(KilterRepository)
