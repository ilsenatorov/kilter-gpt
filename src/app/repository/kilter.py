import logging

from os import name
from typing import Any, Optional, Dict
from src.app.models.generation import ClimbGrade, Feedback, Climb, GenerationParams
from src.app.db.client import supabase_client
from src.app.db.kilter import GenerationMetadata, KilterClimbStats, TableNames, KilterClimb, GeneratedClimbFeedback


# TODO: maybe refactor all db queries to async behavior?
# TODO: extract all dto->model convert methods into KilterDBMapper class?
class KilterRepository:
    difficulty_mapper = {grade.value: 13 + i for i, grade in enumerate(ClimbGrade)}

    def __init__(self):
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.info(f"Inititalized {self.__class__.__name__}")

            self.client = supabase_client

    def save_generation(self, climb_name: str, generation_params: GenerationParams, holds: str) -> str:
        climb_id = self._save_generated_climb(holds, climb_name)
        self._save_generation_params(climb_id, generation_params)
        self._save_climb_stats(climb_id, generation_params)

        return climb_id

    def save_feedback(self, feedback: Feedback) -> None:
        self.logger.debug("Updating feedback...")
        feedback_model = self._feedback_dto_to_model(feedback)

        current_feedback = self.client.table(TableNames.climb_feedback_table).select("*").execute()
        if len(current_feedback.data) == 0:
            self.client.table(TableNames.climb_feedback_table).insert(feedback_model.to_dict()).execute()
        else:
            feedback_model = self._feedback_dto_to_model(feedback)
            self.client.table(TableNames.climb_feedback_table).update(feedback_model.to_dict()).eq("id", feedback.climb_id).execute()

    def _save_generated_climb(self, holds: str, climb_name: str) -> str:
        self.logger.info("Saving generated climb...")
        kilter_climb = KilterClimb(name=climb_name, frames=holds)
        response = self.client.table(TableNames.climbs_table).insert(kilter_climb.to_dict()).execute()
        return response.data[0]["id"]

    def _save_generation_params(self, climb_id: str, generation_params: GenerationParams):
        self.logger.info("Saving generation metadata...")
        gen_params_model = self._generation_params_dto_to_metadata_model(generation_params, climb_id)
        self.client.table(TableNames.generation_metadata_table).insert(gen_params_model.to_dict()).execute()

    def _save_climb_stats(self, climb_id: str, generation_params: GenerationParams):
        self.logger.info("Saving climb stats...")
        gen_params_model = self._generation_params_dto_to_stats_model(generation_params, climb_id)
        self.client.table(TableNames.climb_stats_table).insert(gen_params_model.to_dict()).execute()

    @staticmethod
    def _feedback_dto_to_model(feedback: Feedback) -> GeneratedClimbFeedback:
        return GeneratedClimbFeedback(
            climb_id=feedback.climb_id,
            user_id=feedback.user_id,
            type=feedback.feedback
        )

    @staticmethod
    def _generation_params_dto_to_metadata_model(generation_params: GenerationParams, climb_id: str) -> GenerationMetadata:
        return GenerationMetadata(
            climb_id=climb_id,
            prompt_frames=generation_params.frames,
            temperature=generation_params.temperature,
            top_p_sampling=generation_params.p
        )

    @staticmethod
    def _generation_params_dto_to_stats_model(generation_params: GenerationParams, climb_id: str) -> KilterClimbStats:
        difficulty = KilterRepository.difficulty_mapper[generation_params.grade]
        return KilterClimbStats(
            climb_id=climb_id,
            angle=generation_params.angle,
            difficulty=difficulty
        )
