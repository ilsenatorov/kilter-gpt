import logging
import sqlalchemy

from typing import Any, Optional
from sqlalchemy import sql, update
from app.models.generation import Feedback, Generation
from src.app.db.engine import engine
from src.app.db.generations import GenerationsData


class GenerationsRepository:

    def __init__(self):
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.info(f"Inititalized {self.__class__.__name__}")

            # TODO: declare engine/connection/client from db layer here as class attribute

    def save_generation(self, generation: Generation) -> int | None:
        # TODO: maybe async behavior?
        pass

    def update_feedback(self, feedback: Feedback) -> None:
        # TODO: maybe async behavior?
        pass

    # TODO: close connection method ?
