"""
Module for WEB-server start.
"""

from fastapi import FastAPI

from kiltergpt.app.api.v0 import router as v0_router


def create_app() -> FastAPI:
    app = FastAPI(docs_url="/swagger")
    app.include_router(v0_router, prefix="/api", tags=["v0"])
    return app
