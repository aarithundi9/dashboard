"""CSV-file backend – drop-in replacement for the in-memory Database.

Stores runs and steps in two CSV files inside a configurable data directory.
Data is loaded into memory on startup and flushed to disk on every write,
giving the speed of in-memory access with simple file-based persistence.

Environment variables (all optional):
    CSV_DATA_DIR – directory for the CSV files (default: ./data)
"""
from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from typing import Optional

from models import Run, RunMetadata, RunStatus, Step, StepError, StepStatus, StepType, SystemType

logger = logging.getLogger(__name__)

# CSV column definitions
RUN_COLUMNS = [
    "run_id", "created_at", "updated_at", "status",
    "system_type", "root_step_id", "user_id", "tags",
]

STEP_COLUMNS = [
    "step_id", "run_id", "parent_step_id", "name", "type", "status",
    "started_at", "ended_at", "duration_ms", "tokens_prompt",
    "tokens_completion", "cost_usd", "input", "output",
    "error_message", "error_code", "error_stack",
]


class CsvDatabase:
    """Persistent CSV store matching the Database / DatabricksDatabase interface."""

    def __init__(self, data_dir: str | None = None) -> None:
        self._dir = Path(data_dir or os.getenv("CSV_DATA_DIR", "./data"))
        self._dir.mkdir(parents=True, exist_ok=True)
        self._runs_path = self._dir / "runs.csv"
        self._steps_path = self._dir / "steps.csv"

        # In-memory indexes (same shape as the in-memory Database)
        self.runs: dict[str, Run] = {}
        self.steps: dict[str, Step] = {}

        self._load()
        logger.info(
            "CSV database ready  (%d runs, %d steps)  dir=%s",
            len(self.runs), len(self.steps), self._dir,
        )

    # ── Loading / saving helpers ──────────────────────────────────────────────

    def _load(self) -> None:
        if self._runs_path.exists():
            with open(self._runs_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    run = self._row_to_run(row)
                    self.runs[run.run_id] = run
        if self._steps_path.exists():
            with open(self._steps_path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    step = self._row_to_step(row)
                    self.steps[step.step_id] = step

    def _flush_runs(self) -> None:
        with open(self._runs_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=RUN_COLUMNS)
            writer.writeheader()
            for run in self.runs.values():
                writer.writerow(self._run_to_row(run))

    def _flush_steps(self) -> None:
        with open(self._steps_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=STEP_COLUMNS)
            writer.writeheader()
            for step in self.steps.values():
                writer.writerow(self._step_to_row(step))

    # ── Serialisation helpers ─────────────────────────────────────────────────

    @staticmethod
    def _run_to_row(run: Run) -> dict:
        return {
            "run_id": run.run_id,
            "created_at": run.created_at,
            "updated_at": run.updated_at,
            "status": run.status.value,
            "system_type": run.system_type.value,
            "root_step_id": run.root_step_id or "",
            "user_id": run.metadata.user_id,
            "tags": json.dumps(run.metadata.tags),
        }

    @staticmethod
    def _row_to_run(row: dict) -> Run:
        return Run(
            run_id=row["run_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            status=RunStatus(row["status"]),
            system_type=SystemType(row["system_type"]),
            root_step_id=row["root_step_id"] or None,
            metadata=RunMetadata(
                user_id=row.get("user_id") or "demo",
                tags=json.loads(row["tags"]) if row.get("tags") else ["demo"],
            ),
        )

    @staticmethod
    def _step_to_row(step: Step) -> dict:
        return {
            "step_id": step.step_id,
            "run_id": step.run_id,
            "parent_step_id": step.parent_step_id or "",
            "name": step.name,
            "type": step.type.value,
            "status": step.status.value,
            "started_at": step.started_at,
            "ended_at": step.ended_at or "",
            "duration_ms": step.duration_ms,
            "tokens_prompt": step.tokens_prompt,
            "tokens_completion": step.tokens_completion,
            "cost_usd": step.cost_usd,
            "input": json.dumps(step.input),
            "output": json.dumps(step.output),
            "error_message": step.error.message if step.error else "",
            "error_code": step.error.code if step.error else "",
            "error_stack": step.error.stack if step.error else "",
        }

    @staticmethod
    def _row_to_step(row: dict) -> Step:
        error = None
        if row.get("error_message"):
            error = StepError(
                message=row["error_message"],
                code=row.get("error_code") or None,
                stack=row.get("error_stack") or None,
            )
        return Step(
            step_id=row["step_id"],
            run_id=row["run_id"],
            parent_step_id=row["parent_step_id"] or None,
            name=row["name"],
            type=StepType(row["type"]),
            status=StepStatus(row["status"]),
            started_at=row["started_at"],
            ended_at=row.get("ended_at") or "",
            duration_ms=int(row.get("duration_ms") or 0),
            tokens_prompt=int(row.get("tokens_prompt") or 0),
            tokens_completion=int(row.get("tokens_completion") or 0),
            cost_usd=float(row.get("cost_usd") or 0.0),
            input=json.loads(row["input"]) if row.get("input") else {},
            output=json.loads(row["output"]) if row.get("output") else {},
            error=error,
        )

    # ── Runs ──────────────────────────────────────────────────────────────────

    def create_run(self, run: Run) -> Run:
        self.runs[run.run_id] = run
        self._flush_runs()
        return run

    def get_run(self, run_id: str) -> Optional[Run]:
        return self.runs.get(run_id)

    def list_runs(self, limit: int = 50) -> list[Run]:
        runs = sorted(self.runs.values(), key=lambda r: r.created_at, reverse=True)
        return runs[:limit]

    def update_run(self, run: Run) -> Run:
        self.runs[run.run_id] = run
        self._flush_runs()
        return run

    # ── Steps ─────────────────────────────────────────────────────────────────

    def create_step(self, step: Step) -> Step:
        self.steps[step.step_id] = step
        self._flush_steps()
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
        self._flush_steps()
        return step

    # ── Batch helpers (used by the data generator) ────────────────────────────

    def bulk_insert_runs(self, runs: list[Run]) -> None:
        for r in runs:
            self.runs[r.run_id] = r
        self._flush_runs()

    def bulk_insert_steps(self, steps: list[Step]) -> None:
        for s in steps:
            self.steps[s.step_id] = s
        self._flush_steps()
