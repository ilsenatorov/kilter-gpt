"""Work with generations table data."""

import logging
from typing import Any, Optional

import sqlalchemy
from sqlalchemy import sql, update

import engine



class GenerationsData:
    """Generations table declaration."""

    __tablename__ = "generations"  # type: ignore

    class GenerationsTableColumns:
        ID = "id"
        GENERATED_HOLDS = "generated_holds"
        FEEDBACK = "feedback"
        CREATED = "created"

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Inititalized {self.__class__.__name__}")

        self.connection = engine.connect()

        self._generations_table = sql.table(
            self.__tablename__,
            sql.column(self.GenerationsTableColumns.ID),
            sql.column(self.GenerationsTableColumns.GENERATED_HOLDS),
            sql.column(self.GenerationsTableColumns.FEEDBACK),
            sql.column(self.GenerationsTableColumns.CREATED),
        )

    def save_generation(self, generation: Any) -> int | None:
        values = {
            # TODO: fill with actual data
            "generation": generation
        }

        # returning id here on purpose: we will need it to update records with received feedback if we'd have one
        statement = sql.insert(self._generations_table).values(**values).returning(self._generations_table.c.id)

        try:
            result = self.connection.execute(statement.execution_options(autocommit=True)).fetchone()
            if result is not None:
                return result[0]

            return result
        except Exception as e:
            self.logger.exception(f"Failed sql statement '{statement}', with exception: {e}")

    def update_feedback(self, id: int, feedback: str) -> None:
        values = {
            "feedback": feedback
        }

        statement = update(self._generations_table).values(**values).where(
            self._generations_table.c.id == id  # TODO: test wether it works or work_id is a better way
        )

        try:
            self.connection.execute(statement.execution_options(autocommit=True))
        except Exception as e:
            self.logger.exception(f"Failed sql statement '{statement}', with exception: {e}")


    def close(self):
        self.connection.close()
