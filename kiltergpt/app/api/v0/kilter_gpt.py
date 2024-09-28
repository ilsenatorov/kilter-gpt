"""
API for KilterpGPT service.
"""

import logging
from http import HTTPStatus
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from starlette.responses import JSONResponse

from kiltergpt.app.models.generation import Feedback, GenerationParams, Climb
from kiltergpt.app.service.kilter_gpt import KilterService

from kiltergpt.app.dependencies import get_kilter_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/generate",
    status_code=HTTPStatus.OK,
    description="Generate kilter climb with given params",
)
def generate_route(
    generation_params: GenerationParams,
    kilter_service: Annotated[KilterService, Depends(get_kilter_service)],
) -> Climb:
    return kilter_service.generate_route(generation_params)


@router.put(
    "/feedback",
    status_code=HTTPStatus.OK,
    description="Update feedback for the given climb",
)
def update_feedback(
    feedback: Feedback,
    kilter_service: Annotated[KilterService, Depends(get_kilter_service)],
) -> JSONResponse:
    kilter_service.save_feedback(feedback)
    return JSONResponse("ok")
