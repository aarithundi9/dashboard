"""Databricks Delta Lake backend – drop-in replacement for the in-memory Database.

Reads all connection details from environment variables:
    DATABRICKS_SERVER_HOSTNAME  – e.g. adb-1234567890.azuredatabricks.net
    DATABRICKS_HTTP_PATH        – SQL Warehouse HTTP path
    DATABRICKS_TOKEN            – Personal Access Token or Service Principal secret
    DATABRICKS_CATALOG          – Unity Catalog name (default: hive_metastore)
    DATABRICKS_SCHEMA           – Schema / database name (default: default)
"""
from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from typing import Optional

from databricks import sql as dbsql

from models import Run, RunMetadata, RunStatus, Step, StepError, StepStatus, StepType, SystemType

logger = logging.getLogger(__name__)


class DatabricksDatabase:
    """Persistent Delta Lake store matching the Database interface."""

    def __init__(self) -> None:
        self.server_hostname = os.environ["DATABRICKS_SERVER_HOSTNAME"]
        self.http_path = os.environ["DATABRICKS_HTTP_PATH"]
        self.access_token = os.environ["DATABRICKS_TOKEN"]
        self.catalog = os.getenv("DATABRICKS_CATALOG", "workspace")
        self.schema = os.getenv("DATABRICKS_SCHEMA", "agent_dashboard")
        self._fqn = f"`{self.catalog}`.`{self.schema}`"
        self._ensure_tables()

    # ── Connection helper ─────────────────────────────────────────────────────

    @contextmanager
    def _cursor(self):
        conn = dbsql.connect(
            server_hostname=self.server_hostname,
            http_path=self.http_path,
            access_token=self.access_token,
        )
        try:
            cursor = conn.cursor()
            try:
                yield cursor
                conn.commit()
            finally:
                cursor.close()
        finally:
            conn.close()

    # ── Schema bootstrap ──────────────────────────────────────────────────────

    def _ensure_tables(self) -> None:
        logger.info("Ensuring Databricks Delta tables exist …")
        with self._cursor() as cur:
            cur.execute(f"CREATE SCHEMA IF NOT EXISTS {self.catalog}.{self.schema}")

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._fqn}.agent_runs (
                    run_id        STRING NOT NULL,
                    created_at    STRING,
                    updated_at    STRING,
                    status        STRING,
                    system_type   STRING,
                    root_step_id  STRING,
                    user_id       STRING,
                    tags          STRING
                ) USING DELTA
                TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true')
            """)

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._fqn}.agent_steps (
                    step_id           STRING NOT NULL,
                    run_id            STRING NOT NULL,
                    parent_step_id    STRING,
                    name              STRING,
                    type              STRING,
                    status            STRING,
                    started_at        STRING,
                    ended_at          STRING,
                    duration_ms       BIGINT,
                    tokens_prompt     BIGINT,
                    tokens_completion BIGINT,
                    cost_usd          DOUBLE,
                    input             STRING,
                    output            STRING,
                    error_message     STRING,
                    error_code        STRING,
                    error_stack       STRING
                ) USING DELTA
                TBLPROPERTIES ('delta.autoOptimize.optimizeWrite' = 'true')
            """)
        logger.info("Delta tables ready.")

    # ── Serialisation helpers ─────────────────────────────────────────────────

    @staticmethod
    def _run_to_row(run: Run) -> tuple:
        return (
            run.run_id,
            run.created_at,
            run.updated_at,
            run.status.value,
            run.system_type.value,
            run.root_step_id,
            run.metadata.user_id,
            json.dumps(run.metadata.tags),
        )

    @staticmethod
    def _row_to_run(row) -> Run:
        run_id, created_at, updated_at, status, system_type, root_step_id, user_id, tags_json = row
        return Run(
            run_id=run_id,
            created_at=created_at,
            updated_at=updated_at,
            status=RunStatus(status),
            system_type=SystemType(system_type),
            root_step_id=root_step_id,
            metadata=RunMetadata(
                user_id=user_id or "demo",
                tags=json.loads(tags_json) if tags_json else ["demo"],
            ),
        )

    @staticmethod
    def _step_to_row(step: Step) -> tuple:
        error_msg = error_code = error_stack = None
        if step.error:
            error_msg = step.error.message
            error_code = step.error.code
            error_stack = step.error.stack
        return (
            step.step_id,
            step.run_id,
            step.parent_step_id,
            step.name,
            step.type.value,
            step.status.value,
            step.started_at,
            step.ended_at,
            step.duration_ms,
            step.tokens_prompt,
            step.tokens_completion,
            step.cost_usd,
            json.dumps(step.input),
            json.dumps(step.output),
            error_msg,
            error_code,
            error_stack,
        )

    @staticmethod
    def _row_to_step(row) -> Step:
        (
            step_id, run_id, parent_step_id, name, type_, status,
            started_at, ended_at, duration_ms, tokens_prompt, tokens_completion,
            cost_usd, input_json, output_json, error_msg, error_code, error_stack,
        ) = row

        error = None
        if error_msg:
            error = StepError(message=error_msg, code=error_code, stack=error_stack)

        return Step(
            step_id=step_id,
            run_id=run_id,
            parent_step_id=parent_step_id,
            name=name,
            type=StepType(type_),
            status=StepStatus(status),
            started_at=started_at,
            ended_at=ended_at or "",
            duration_ms=int(duration_ms or 0),
            tokens_prompt=int(tokens_prompt or 0),
            tokens_completion=int(tokens_completion or 0),
            cost_usd=float(cost_usd or 0.0),
            input=json.loads(input_json) if input_json else {},
            output=json.loads(output_json) if output_json else {},
            error=error,
        )

    # ── Runs ──────────────────────────────────────────────────────────────────

    def create_run(self, run: Run) -> Run:
        with self._cursor() as cur:
            cur.execute(
                f"INSERT INTO {self._fqn}.agent_runs VALUES (?,?,?,?,?,?,?,?)",
                self._run_to_row(run),
            )
        return run

    def get_run(self, run_id: str) -> Optional[Run]:
        with self._cursor() as cur:
            cur.execute(
                f"SELECT * FROM {self._fqn}.agent_runs WHERE run_id = ?",
                (run_id,),
            )
            row = cur.fetchone()
        return self._row_to_run(row) if row else None

    def list_runs(self, limit: int = 50) -> list[Run]:
        with self._cursor() as cur:
            cur.execute(
                f"SELECT * FROM {self._fqn}.agent_runs ORDER BY created_at DESC LIMIT {int(limit)}"
            )
            rows = cur.fetchall()
        return [self._row_to_run(r) for r in rows]

    def update_run(self, run: Run) -> Run:
        with self._cursor() as cur:
            cur.execute(
                f"""UPDATE {self._fqn}.agent_runs SET
                    updated_at   = ?,
                    status       = ?,
                    root_step_id = ?
                WHERE run_id = ?""",
                (run.updated_at, run.status.value, run.root_step_id, run.run_id),
            )
        return run

    # ── Steps ─────────────────────────────────────────────────────────────────

    def create_step(self, step: Step) -> Step:
        with self._cursor() as cur:
            cur.execute(
                f"INSERT INTO {self._fqn}.agent_steps VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                self._step_to_row(step),
            )
        return step

    def get_step(self, step_id: str) -> Optional[Step]:
        with self._cursor() as cur:
            cur.execute(
                f"SELECT * FROM {self._fqn}.agent_steps WHERE step_id = ?",
                (step_id,),
            )
            row = cur.fetchone()
        return self._row_to_step(row) if row else None

    def get_steps_for_run(self, run_id: str) -> list[Step]:
        with self._cursor() as cur:
            cur.execute(
                f"SELECT * FROM {self._fqn}.agent_steps WHERE run_id = ? ORDER BY started_at",
                (run_id,),
            )
            rows = cur.fetchall()
        return [self._row_to_step(r) for r in rows]

    def update_step(self, step: Step) -> Step:
        with self._cursor() as cur:
            cur.execute(
                f"""UPDATE {self._fqn}.agent_steps SET
                    status            = ?,
                    ended_at          = ?,
                    duration_ms       = ?,
                    output            = ?,
                    error_message     = ?,
                    error_code        = ?,
                    error_stack       = ?
                WHERE step_id = ?""",
                (
                    step.status.value,
                    step.ended_at,
                    step.duration_ms,
                    json.dumps(step.output),
                    step.error.message if step.error else None,
                    step.error.code if step.error else None,
                    step.error.stack if step.error else None,
                    step.step_id,
                ),
            )
        return step

    # ── Batch helpers (used by the data generator) ────────────────────────────

    def bulk_insert_runs(self, runs: list[Run]) -> None:
        """Insert many runs in one round-trip."""
        if not runs:
            return
        with self._cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self._fqn}.agent_runs VALUES (?,?,?,?,?,?,?,?)",
                [self._run_to_row(r) for r in runs],
            )

    def bulk_insert_steps(self, steps: list[Step]) -> None:
        """Insert many steps in one round-trip."""
        if not steps:
            return
        with self._cursor() as cur:
            cur.executemany(
                f"INSERT INTO {self._fqn}.agent_steps VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [self._step_to_row(s) for s in steps],
            )
