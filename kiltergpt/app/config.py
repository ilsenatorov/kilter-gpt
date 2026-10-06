"""Service configuration"""

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ENV_FILE = os.environ.get("ENV_FILE") or ".env"


class Settings(BaseSettings):
    BIND_IP: str = "127.0.0.1"
    BIND_PORT: int = 8000
    HOSTNAME: str = "default"
    BACKEND_CORS_ORIGINS: str = "*"

    # ==== Logging settings ====
    LOGGING_LEVEL: str = "DEBUG"
    # TODO: provide "LOGGING_PATH" with logging config

    # ==== Database settings ====
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""

    # ==== Model settings ==== # TODO currently unused
    MODEL_CHECKPOINT: str = str(Path(__file__).resolve().parents[2] / "artifacts/model-tamkrd8d:v0/model.ckpt")

    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")


settings = Settings()
