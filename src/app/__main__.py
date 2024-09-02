"""
Module for local start of kilter-gpt service.
"""

import uvicorn

from src.app.config import settings

if __name__ == "__main__":
    uvicorn.run(
        "src.app.main:create_app",
        host=settings.BIND_IP,
        port=settings.BIND_PORT,
        reload=True,
    )
