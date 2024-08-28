"""Service configuration"""

import os

from pydantic_settings import BaseSettings

ENV_FILE = os.environ.get("ENV_FILE") or ".env"


class Settings(BaseSettings):
    # ==== Logging settings ====
    LOGGING_LEVEL: str = "DEBUG"
    # TODO: provide "LOGGING_PATH" with logging config

    # ==== Database settings ====
    DB_URL: str = ""
    DB_ECHO: bool = True
    DB_DEFAULT_SCHEMA: str = "service"

    class Config:
        env_file = ENV_FILE
        env_file_encoding = "utf-8"


settings = Settings()
