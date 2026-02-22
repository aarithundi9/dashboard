"""Database factory – returns the configured storage backend.

Set DATABASE_BACKEND in your .env (or environment):
    csv        – CSV files in ./data  (default, persistent, fast)
    databricks – Databricks Delta Lake
    memory     – ephemeral in-memory store (dev / CI)
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()  # load .env if present

from models import Run, Step

logger = logging.getLogger(__name__)


class Database:
    """Simple in-memory store (development / CI fallback)."""

    def __init__(self) -> None:
        self.runs: dict[str, Run] = {}
        self.steps: dict[str, Step] = {}  # keyed by step_id

    # ── Runs ─────────────────────────────────────────────────────────────

    def create_run(self, run: Run) -> Run:
        self.runs[run.run_id] = run
        return run

    def get_run(self, run_id: str) -> Optional[Run]:
        return self.runs.get(run_id)

    def list_runs(self, limit: int = 50) -> list[Run]:
        runs = sorted(self.runs.values(), key=lambda r: r.created_at, reverse=True)
        return runs[:limit]

    def update_run(self, run: Run) -> Run:
        self.runs[run.run_id] = run
        return run

    # ── Steps ────────────────────────────────────────────────────────────

    def create_step(self, step: Step) -> Step:
        self.steps[step.step_id] = step
        return step

    def get_step(self, step_id: str) -> Optional[Step]:
        return self.steps.get(step_id)

    def get_steps_for_run(self, run_id: str) -> list[Step]:
        return sorted(
            [s for s in self.steps.values() if s.run_id == run_id],
            key=lambda s: s.started_at,
        )

    def update_step(self, step: Step) -> Step:
        self.steps[step.step_id] = step
        return step

    # ── Batch helpers (no-op stubs so the generator works with either backend)
    def bulk_insert_runs(self, runs: list[Run]) -> None:
        for r in runs:
            self.runs[r.run_id] = r

    def bulk_insert_steps(self, steps: list[Step]) -> None:
        for s in steps:
            self.steps[s.step_id] = s


def _build_db():
    backend = os.getenv("DATABASE_BACKEND", "csv").lower()
    if backend == "databricks":
        try:
            from databricks_database import DatabricksDatabase
            logger.info("Using Databricks Delta Lake backend.")
            return DatabricksDatabase()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to initialise Databricks backend (%s). "
                "Falling back to CSV store.",
                exc,
            )
            backend = "csv"  # fall through to CSV

    if backend == "csv":
        try:
            from csv_database import CsvDatabase
            logger.info("Using CSV file backend.")
            return CsvDatabase()
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "Failed to initialise CSV backend (%s). "
                "Falling back to in-memory store.",
                exc,
            )

    logger.info("Using in-memory database backend.")
    return Database()


# Singleton – imported everywhere as `from database import db`
db = _build_db()
