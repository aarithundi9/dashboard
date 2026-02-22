"""UAOP Demo – FastAPI application."""
from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware

from models import (
    Run, Step, CreateRunRequest, CreateStepRequest,
    RunStatus, StepStatus, StepType, RunMetadata,
)
from database import db
from websocket_manager import manager
from simulator import run_simulation
from scenarios import SCENARIOS, SCENARIO_LABELS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("UAOP API starting up")
    yield
    logger.info("UAOP API shutting down")


app = FastAPI(
    title="UAOP Demo API",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS – allow the Next.js dev server (ports 3000 and 3001)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://127.0.0.1:3000", "http://127.0.0.1:3001", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── REST Endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "uaop-api"}


@app.get("/api/scenarios")
async def list_scenarios():
    """Return available demo scenarios."""
    return {
        "scenarios": [
            {"id": k, "label": v}
            for k, v in SCENARIO_LABELS.items()
        ]
    }


@app.post("/api/runs")
async def create_run(req: CreateRunRequest):
    """Create a new run and optionally start a simulation."""
    run = Run(
        system_type=req.system_type,
        metadata=req.metadata or RunMetadata(),
    )
    db.create_run(run)
    logger.info(f"Created run {run.run_id} (scenario={req.scenario})")

    # Start simulation in background if a scenario is provided
    if req.scenario and req.scenario in SCENARIOS:
        asyncio.create_task(run_simulation(run.run_id, req.scenario))

    return run.model_dump()


@app.get("/api/runs")
async def list_runs(limit: int = Query(50, ge=1, le=200)):
    """List recent runs."""
    runs = db.list_runs(limit)
    return {"runs": [r.model_dump() for r in runs]}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str):
    """Get a single run by ID."""
    run = db.get_run(run_id)
    if not run:
        return {"error": "Run not found"}, 404
    return run.model_dump()


@app.get("/api/runs/{run_id}/steps")
async def get_run_steps(run_id: str):
    """Get all steps for a run."""
    steps = db.get_steps_for_run(run_id)
    return {"steps": [s.model_dump() for s in steps]}


@app.post("/api/steps")
async def create_step(req: CreateStepRequest):
    """Create a step manually (for non-simulated use)."""
    from datetime import datetime, timezone

    step = Step(
        run_id=req.run_id,
        parent_step_id=req.parent_step_id,
        name=req.name,
        type=StepType(req.type),
        input=req.input,
    )
    db.create_step(step)

    await manager.broadcast(req.run_id, {
        "type": "step_update",
        "step": step.model_dump(),
    })

    return step.model_dump()


# ── WebSocket ──────────────────────────────────────────────────────────────────

@app.websocket("/ws/runs/{run_id}")
async def websocket_endpoint(ws: WebSocket, run_id: str):
    """Subscribe to real-time updates for a specific run."""
    await manager.connect(run_id, ws)
    try:
        while True:
            # Keep connection alive; client can send ping/pong
            data = await ws.receive_text()
            # Echo back as acknowledgment
            if data == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(run_id, ws)
    except Exception:
        manager.disconnect(run_id, ws)
