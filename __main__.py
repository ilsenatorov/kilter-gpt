"""
Module for local start of kilter-gpt service.
"""

import uvicorn

from kiltergpt.app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "kiltergpt.app.main:create_app",
        host=settings.BIND_IP,
        port=settings.BIND_PORT,
        reload=True,
    )
