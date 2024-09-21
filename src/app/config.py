"""Service configuration"""

import os

from pydantic_settings import BaseSettings

ENV_FILE = os.environ.get("ENV_FILE") or ".env"


class Settings(BaseSettings):
    BIND_IP: str = "0.0.0.0"
    BIND_PORT: int = 8000
    HOSTNAME: str = "default"
    BACKEND_CORS_ORIGINS: str = "*"

    # ==== Logging settings ====
    LOGGING_LEVEL: str = "DEBUG"
    # TODO: provide "LOGGING_PATH" with logging config

    # ==== Database settings ====
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""

    # ==== Model settings ====
    WANDB_MODEL_NAME: str = ""

    class Config:
        env_file = ENV_FILE
        env_file_encoding = "utf-8"


settings = Settings()
