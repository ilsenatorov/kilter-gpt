"""
Setup DB connection.
"""

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from src.app.config import settings


def get_pg_engine(
    database_host: str = "127.0.0.1",
    database_port: str = "5429",
    database_user: str = "postgres",
    database_password: str = "postgres",
    database_name: str = "postgres",
) -> Engine:
    db_string = f"postgresql://{database_user}:{database_password}@{database_host}:{database_port}/{database_name}"

    return create_engine(db_string)


def create_db_engine() -> Engine:
    return get_pg_engine(
        database_host=settings.DB_URL,
        database_port=settings.DB_PORT,
        database_name=settings.DB_NAME,
        database_user=settings.DB_USER,
        database_password=settings.DB_PASSWORD,
    )


engine = create_db_engine()
