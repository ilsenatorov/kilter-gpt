import logging

from typing import Any, Optional, Dict
from app.models.generation import Feedback, Generation
from src.app.db.client import supabase_client
from src.app.db.generations import GenerationsData


class GenerationsRepository:

    def __init__(self):
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.info(f"Inititalized {self.__class__.__name__}")

            self.client = supabase_client

    def save_generation(self, generation: Generation) -> int | None:
        # TODO: maybe async behavior?
        values = self._generation_to_supabase_dict(generation)
        self.client.table(GenerationsData.table_name).insert(values)  # TODO: come up with id returning

    def update_feedback(self, feedback: Feedback) -> None:
        # TODO: maybe async behavior?
        values = self._feedback_to_supabase_dict(feedback)
        self.client.table(GenerationsData.table_name).update(values).eq("id", feedback.climb_id)

    @staticmethod
    def _generation_to_supabase_dict(generation: Generation) -> Dict[str, Any]:
        return {}

    @staticmethod
    def _feedback_to_supabase_dict(feedback: Feedback) -> Dict[str, bool | None]:
        return {"feedback": feedback.feedback}
