"""
Module for WEB-server start.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.app.api.v0 import router as v0_router
from src.app.config import settings
from src.app.dependencies import get_kilter_service

logger = logging.getLogger(__name__)


def setup_routers(app: FastAPI) -> None:
    """
    Router initialization.
    """
    app.include_router(router=v0_router, prefix="/api", tags=["v0"])


def create_app(*args: Any, **kwargs: Any) -> FastAPI:
    """
    Creates FastAPI web-server.
    """
    container = get_kilter_service()
    app = FastAPI(docs_url="/swagger")
    setup_routers(app)
    return app
