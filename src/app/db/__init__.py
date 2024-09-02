"""Place here all db models."""
from sqlalchemy.orm import configure_mappers

from .engine import engine
from .generations import GenerationsData

configure_mappers()
