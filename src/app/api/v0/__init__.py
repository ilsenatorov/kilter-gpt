"""Пример сервиса."""

from fastapi import APIRouter

from .kilter_gpt import router as kilter_router

router = APIRouter(prefix="/kilter_gpt")
router.include_router(router=kilter_router)
