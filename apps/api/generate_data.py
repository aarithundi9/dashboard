"""Batch data generator – writes simulated agent automation data to CSV files.

Creates hundreds of historical runs across multiple agent systems and scenarios,
with realistic performance variation to support PoC insights and dashboards.

Usage (writes to ./data/*.csv):
    python generate_data.py --runs 300 --days 30

Usage (dry-run to stdout, no files written):
    python generate_data.py --dry-run --runs 20

The generated dataset is designed to surface interesting PoC insights:
  - Claude adoption growing over time vs OpenAI / OpenClaw
  - Per-scenario cost & duration differences across agent types
  - Failure rate and retry patterns
  - Token usage trends
  - P50 / P95 latency by scenario
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(format="%(levelname)s  %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Agent profiles
# Each key maps to a SystemType value. Profiles define realistic performance
# characteristics so the generated data tells a useful comparative story.
# ─────────────────────────────────────────────────────────────────────────────

AGENT_PROFILES: dict[str, dict] = {
    "claude": {
        # Anthropic Claude 3.5 Sonnet – low cost, fast, high reliability
        "token_mult": 0.95,        # slightly fewer tokens (concise model)
        "duration_mult": 0.82,     # 18 % faster than baseline
        "prompt_cost_per_k": 0.003,
        "completion_cost_per_k": 0.015,
        "success_rate": 0.94,      # 94 % of runs succeed
        "retry_rate": 0.08,        # 8 % chance a failed step gets retried
        # Weight in the time-weighted distribution:
        # early in the window claude = 15 %, late in window claude = 45 %
        # (simulates growing adoption)
        "weight_early": 0.15,
        "weight_late": 0.45,
        "label": "claude",
    },
    "openai": {
        # OpenAI GPT-4 – higher cost, fast, very reliable
        "token_mult": 1.15,
        "duration_mult": 0.88,
        "prompt_cost_per_k": 0.030,
        "completion_cost_per_k": 0.060,
        "success_rate": 0.92,
        "retry_rate": 0.09,
        "weight_early": 0.40,
        "weight_late": 0.28,
        "label": "other",   # maps to SystemType.other (treat as generic openai)
    },
    "openclaw": {
        # Hypothetical internally-hosted OpenClaw – moderate cost, slower
        "token_mult": 0.90,
        "duration_mult": 1.25,
        "prompt_cost_per_k": 0.008,
        "completion_cost_per_k": 0.024,
        "success_rate": 0.86,
        "retry_rate": 0.16,
        "weight_early": 0.30,
        "weight_late": 0.17,
        "label": "openclaw",
    },
    "mock": {
        # Simulated / baseline mock agent
        "token_mult": 0.75,
        "duration_mult": 0.45,
        "prompt_cost_per_k": 0.001,
        "completion_cost_per_k": 0.002,
        "success_rate": 0.97,
        "retry_rate": 0.03,
        "weight_early": 0.15,
        "weight_late": 0.10,
        "label": "mock",
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Scenario step templates (simplified blueprints)
# Format: list of step dicts traversed depth-first to build a run's step tree.
# Each step: {name, type, tokens_p, tokens_c, duration_ms, fail_p, children}
# ─────────────────────────────────────────────────────────────────────────────

def _s(name, type_, tokens_p=0, tokens_c=0, duration_ms=1000, fail_p=0.0, children=None):
    return {
        "name": name, "type": type_,
        "tokens_p": tokens_p, "tokens_c": tokens_c,
        "duration_ms": duration_ms, "fail_p": fail_p,
        "children": children or [],
    }


SCENARIO_TEMPLATES: dict[str, dict] = {
    "flight_booking": _s("Flight Booking Agent", "plan", 150, 80, 320, children=[
        _s("Search Flights", "tool", 0, 0, 2340, children=[
            _s("Analyze Flight Options", "llm", 820, 340, 1850, children=[
                _s("Book Selected Flight", "tool", 0, 0, 1200, children=[
                    _s("Process Payment", "tool", 0, 0, 3200, fail_p=0.35, children=[
                        _s("Retry Payment", "tool", 0, 0, 2100, children=[
                            _s("Generate Confirmation", "llm", 650, 280, 1400, children=[
                                _s("Final Output", "final", 100, 50, 120),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ]),

    "research_summarizer": _s("Research Summarizer Agent", "plan", 200, 120, 280, children=[
        _s("Web Search: Overview", "tool", 0, 0, 1850, children=[
            _s("Extract Key Facts", "llm", 1400, 450, 1200),
        ]),
        _s("Web Search: Papers", "tool", 0, 0, 2100, children=[
            _s("Extract Paper Highlights", "llm", 1800, 600, 1600),
        ]),
        _s("Web Search: Industry", "tool", 0, 0, 1950, children=[
            _s("Synthesize Findings", "llm", 2400, 900, 2800, children=[
                _s("Write Final Summary", "llm", 1800, 1200, 3200, children=[
                    _s("Final Output", "final", 50, 30, 80),
                ]),
            ]),
        ]),
    ]),

    "code_assistant": _s("Code Debug Assistant", "plan", 180, 90, 350, children=[
        _s("Read Source File", "tool", 0, 0, 180),
        _s("Read Test File", "tool", 0, 0, 150, children=[
            _s("Analyze Error Pattern", "llm", 1200, 450, 2100, children=[
                _s("Search Documentation", "tool", 0, 0, 1400),
                _s("Search Codebase", "tool", 0, 0, 980, children=[
                    _s("Generate Fix", "llm", 1500, 380, 1800, children=[
                        _s("Apply Patch", "tool", 0, 0, 220, children=[
                            _s("Run Tests", "tool", 0, 0, 3400, children=[
                                _s("Format Response", "llm", 600, 200, 900, children=[
                                    _s("Final Output", "final", 80, 40, 60),
                                ]),
                            ]),
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ]),

    "customer_support": _s("Customer Support Agent", "plan", 160, 85, 290, children=[
        _s("Classify Intent", "llm", 450, 120, 1100, children=[
            _s("Retrieve KB Articles", "tool", 0, 0, 850),
            _s("Search Customer History", "tool", 0, 0, 620),
            _s("Sentiment Analysis", "llm", 350, 90, 780, children=[
                _s("Draft Response", "llm", 1600, 550, 2200, children=[
                    _s("Quality Check", "llm", 900, 180, 1400, fail_p=0.05, children=[
                        _s("Final Output", "final", 60, 30, 90),
                    ]),
                ]),
            ]),
        ]),
    ]),

    "simple_happy_path": _s("Simple Task Agent", "plan", 100, 50, 200, children=[
        _s("API Lookup", "tool", 0, 0, 1100, children=[
            _s("Format Response", "llm", 380, 150, 1400, children=[
                _s("Final Output", "final", 50, 25, 50),
            ]),
        ]),
    ]),

    # ── Additional scenarios for richer PoC data ──────────────────────────────

    "data_analysis": _s("Data Analysis Agent", "plan", 200, 100, 400, children=[
        _s("Load Dataset", "tool", 0, 0, 1200),
        _s("Clean & Validate Data", "tool", 0, 0, 2800, fail_p=0.08, children=[
            _s("Statistical Summary", "llm", 1800, 600, 2400, children=[
                _s("Detect Anomalies", "llm", 2200, 800, 3100, children=[
                    _s("Generate Charts", "tool", 0, 0, 1800),
                    _s("Write Analysis Report", "llm", 2800, 1400, 4200, children=[
                        _s("Final Output", "final", 120, 60, 100),
                    ]),
                ]),
            ]),
        ]),
    ]),

    "email_drafting": _s("Email Drafting Agent", "plan", 120, 60, 180, children=[
        _s("Retrieve Context", "tool", 0, 0, 600),
        _s("Classify Request Type", "llm", 320, 80, 850, children=[
            _s("Draft Email", "llm", 1200, 500, 2000, children=[
                _s("Tone Check", "llm", 600, 120, 900, children=[
                    _s("Proofread", "llm", 800, 150, 1100, children=[
                        _s("Final Output", "final", 40, 20, 60),
                    ]),
                ]),
            ]),
        ]),
    ]),

    "sql_query_generation": _s("SQL Query Agent", "plan", 140, 70, 250, children=[
        _s("Parse Natural Language Request", "llm", 600, 200, 1200, children=[
            _s("Fetch Schema Metadata", "tool", 0, 0, 400),
            _s("Generate SQL Query", "llm", 1400, 600, 2200, fail_p=0.12, children=[
                _s("Validate Query", "tool", 0, 0, 300, children=[
                    _s("Execute Query", "tool", 0, 0, 2100, children=[
                        _s("Format Results", "llm", 900, 350, 1500, children=[
                            _s("Final Output", "final", 60, 30, 70),
                        ]),
                    ]),
                ]),
            ]),
        ]),
    ]),
}

SCENARIO_KEYS = list(SCENARIO_TEMPLATES.keys())

# Scenario difficulty multiplier: some scenarios are inherently harder for some
# agent types. Values > 1 increase failure probability; < 1 decreases it.
SCENARIO_AGENT_DIFFICULTY: dict[str, dict[str, float]] = {
    "flight_booking":      {"claude": 0.8,  "openai": 0.9,  "openclaw": 1.2, "mock": 0.7},
    "research_summarizer": {"claude": 0.7,  "openai": 0.8,  "openclaw": 1.1, "mock": 0.6},
    "code_assistant":      {"claude": 0.9,  "openai": 0.7,  "openclaw": 1.3, "mock": 0.8},
    "customer_support":    {"claude": 0.8,  "openai": 0.9,  "openclaw": 1.0, "mock": 0.7},
    "simple_happy_path":   {"claude": 0.5,  "openai": 0.5,  "openclaw": 0.7, "mock": 0.4},
    "data_analysis":       {"claude": 0.9,  "openai": 0.8,  "openclaw": 1.4, "mock": 0.8},
    "email_drafting":      {"claude": 0.6,  "openai": 0.7,  "openclaw": 0.9, "mock": 0.6},
    "sql_query_generation":{"claude": 0.9,  "openai": 0.8,  "openclaw": 1.5, "mock": 0.9},
}


# ─────────────────────────────────────────────────────────────────────────────
# Core generation logic
# ─────────────────────────────────────────────────────────────────────────────

def _jitter(value: float, pct: float = 0.20) -> float:
    """Return value ± pct uniformly."""
    return value * random.uniform(1 - pct, 1 + pct)


def _compute_cost(tokens_p: int, tokens_c: int, profile: dict) -> float:
    return round(
        (tokens_p / 1_000) * profile["prompt_cost_per_k"]
        + (tokens_c / 1_000) * profile["completion_cost_per_k"],
        6,
    )


def _walk_template(
    template: dict,
    run_id: str,
    profile: dict,
    scenario: str,
    ts: datetime,
    parent_step_id: Optional[str],
    overall_fail: bool,
    difficulty: float,
) -> tuple[list[dict], Optional[str], datetime]:
    """Recursively walk a scenario template and produce a flat list of step dicts.

    Returns (step_rows, root_step_id, final_ts).
    """
    step_rows: list[dict] = []
    root_step_id: Optional[str] = None

    def walk(node: dict, parent_id: Optional[str], current_ts: datetime, is_root: bool) -> tuple[Optional[str], datetime]:
        nonlocal root_step_id

        step_id = str(uuid.uuid4())
        if is_root:
            root_step_id = step_id

        tok_p = max(0, int(_jitter(node["tokens_p"] * profile["token_mult"])))
        tok_c = max(0, int(_jitter(node["tokens_c"] * profile["token_mult"])))
        dur_ms = max(50, int(_jitter(node["duration_ms"] * profile["duration_mult"])))
        cost = _compute_cost(tok_p, tok_c, profile)

        started_at = current_ts
        ended_at = current_ts + timedelta(milliseconds=dur_ms)

        # Determine step failure
        effective_fail_p = node["fail_p"] * difficulty
        step_failed = overall_fail and (random.random() < effective_fail_p)

        step_type = node["type"]
        if step_failed:
            step_status = "failed"
            step_type = "error"
        else:
            step_status = "completed"

        step_rows.append({
            "step_id": step_id,
            "run_id": run_id,
            "parent_step_id": parent_id,
            "name": node["name"],
            "type": step_type,
            "status": step_status,
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_ms": dur_ms,
            "tokens_prompt": tok_p,
            "tokens_completion": tok_c,
            "cost_usd": cost,
            "input": json.dumps({"task": node["name"]}),
            "output": json.dumps({}) if step_failed else json.dumps({"result": f"{node['name']} completed"}),
            "error_message": f"{node['name']} failed: tool returned non-200 status" if step_failed else None,
            "error_code": "TOOL_ERROR" if step_failed else None,
            "error_stack": None,
        })

        next_ts = ended_at

        if not step_failed:
            for child in node["children"]:
                _, next_ts = walk(child, step_id, next_ts, is_root=False)

        return step_id, next_ts

    _, final_ts = walk(template, parent_step_id, ts, is_root=True)
    return step_rows, root_step_id, final_ts


def _pick_agent(frac: float) -> str:
    """Pick an agent type. frac ∈ [0,1] represents position in the time window
    (0 = earliest run, 1 = latest run). Claude weight increases over time."""
    agents = list(AGENT_PROFILES.keys())
    weights = [
        AGENT_PROFILES[a]["weight_early"] * (1 - frac) + AGENT_PROFILES[a]["weight_late"] * frac
        for a in agents
    ]
    return random.choices(agents, weights=weights, k=1)[0]


def _random_ts(base: datetime, days: int, frac: float) -> datetime:
    """Pick a random timestamp biased toward frac position in the range."""
    day_offset = int(frac * days)
    # Add some noise around the biased day
    day_offset = max(0, min(days - 1, day_offset + random.randint(-2, 2)))
    # Weekday bias: more runs Mon-Fri
    candidate = base + timedelta(days=day_offset)
    # Random time of day, bias toward business hours (8 am – 8 pm)
    hour = random.choices(range(24), weights=[
        1,1,1,1,1,1,1,2, 4,5,5,5, 5,5,5,5, 4,4,3,3, 2,2,1,1
    ])[0]
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return candidate.replace(hour=hour, minute=minute, second=second, tzinfo=timezone.utc)


def generate_run_batch(
    n_runs: int,
    days: int,
) -> tuple[list[dict], list[dict]]:
    """Generate n_runs complete runs and return (run_rows, step_rows)."""

    now = datetime.now(timezone.utc)
    base = now - timedelta(days=days)

    run_rows: list[dict] = []
    step_rows: list[dict] = []

    fracs = sorted([random.random() for _ in range(n_runs)])

    for i, frac in enumerate(fracs):
        agent_key = _pick_agent(frac)
        profile = AGENT_PROFILES[agent_key]
        scenario = random.choice(SCENARIO_KEYS)
        template = SCENARIO_TEMPLATES[scenario]
        difficulty = SCENARIO_AGENT_DIFFICULTY.get(scenario, {}).get(agent_key, 1.0)

        run_id = str(uuid.uuid4())
        ts = _random_ts(base, days, frac)

        # Decide whether this run ultimately succeeds
        run_fails = random.random() > profile["success_rate"]

        steps, root_step_id, final_ts = _walk_template(
            template=template,
            run_id=run_id,
            profile=profile,
            scenario=scenario,
            ts=ts,
            parent_step_id=None,
            overall_fail=run_fails,
            difficulty=difficulty,
        )

        run_status = "failed" if run_fails else "completed"

        # Metadata tags include scenario name so the UI/dashboard can filter
        tags = json.dumps([scenario, agent_key, "generated"])

        run_rows.append({
            "run_id": run_id,
            "created_at": ts.isoformat(),
            "updated_at": final_ts.isoformat(),
            "status": run_status,
            "system_type": profile["label"],
            "root_step_id": root_step_id,
            "user_id": f"poc-user-{random.randint(1, 10):02d}",
            "tags": tags,
        })
        step_rows.extend(steps)

        if (i + 1) % 50 == 0:
            logger.info("  generated %d / %d runs …", i + 1, n_runs)

    return run_rows, step_rows


# ─────────────────────────────────────────────────────────────────────────────
# CSV insertion
# ─────────────────────────────────────────────────────────────────────────────

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


def _insert_to_csv(run_rows: list[dict], step_rows: list[dict], data_dir: str = "./data") -> None:
    import csv
    from pathlib import Path

    out = Path(data_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs_path = out / "runs.csv"
    steps_path = out / "steps.csv"

    logger.info("Writing %d runs to %s …", len(run_rows), runs_path)
    with open(runs_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RUN_COLUMNS)
        writer.writeheader()
        for r in run_rows:
            writer.writerow({
                "run_id": r["run_id"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
                "status": r["status"],
                "system_type": r["system_type"],
                "root_step_id": r["root_step_id"] or "",
                "user_id": r["user_id"],
                "tags": r["tags"],
            })

    logger.info("Writing %d steps to %s …", len(step_rows), steps_path)
    with open(steps_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=STEP_COLUMNS)
        writer.writeheader()
        for s in step_rows:
            writer.writerow({
                "step_id": s["step_id"],
                "run_id": s["run_id"],
                "parent_step_id": s["parent_step_id"] or "",
                "name": s["name"],
                "type": s["type"],
                "status": s["status"],
                "started_at": s["started_at"],
                "ended_at": s["ended_at"] or "",
                "duration_ms": s["duration_ms"],
                "tokens_prompt": s["tokens_prompt"],
                "tokens_completion": s["tokens_completion"],
                "cost_usd": s["cost_usd"],
                "input": s["input"],
                "output": s["output"],
                "error_message": s["error_message"] or "",
                "error_code": s["error_code"] or "",
                "error_stack": s["error_stack"] or "",
            })

    logger.info("CSV write complete  →  %s", out.resolve())


def _print_dry_run_summary(run_rows: list[dict], step_rows: list[dict]) -> None:
    """Print a quick stats summary without writing to Databricks."""
    from collections import Counter

    agent_counts = Counter(r["status"] + "/" + r["system_type"] for r in run_rows)
    scenario_counts: Counter = Counter()
    for r in run_rows:
        tags = json.loads(r["tags"])
        if tags:
            scenario_counts[tags[0]] += 1

    total_cost = sum(s["cost_usd"] for s in step_rows)
    total_tokens = sum(s["tokens_prompt"] + s["tokens_completion"] for s in step_rows)
    succeeded = sum(1 for r in run_rows if r["status"] == "completed")
    failed = sum(1 for r in run_rows if r["status"] == "failed")

    print("\n=== Dry-run summary ===")
    print(f"  Total runs   : {len(run_rows):,}")
    print(f"  Total steps  : {len(step_rows):,}")
    print(f"  Succeeded    : {succeeded:,} ({100*succeeded//len(run_rows)} %)")
    print(f"  Failed       : {failed:,} ({100*failed//len(run_rows)} %)")
    print(f"  Total cost   : ${total_cost:,.4f}")
    print(f"  Total tokens : {total_tokens:,}")
    print("\n  By agent / status:")
    for k, v in sorted(agent_counts.items()):
        print(f"    {k:<30} {v:>5}")
    print("\n  By scenario:")
    for k, v in sorted(scenario_counts.items()):
        print(f"    {k:<35} {v:>5}")
    print()

    print("  Sample run row:")
    print(json.dumps(run_rows[0], indent=4))
    print("  Sample step row:")
    print(json.dumps(step_rows[0], indent=4))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simulated agent data (writes to CSV)")
    parser.add_argument("--runs", type=int, default=300, help="Number of runs to generate (default: 300)")
    parser.add_argument("--days", type=int, default=30, help="Time window in days (default: 30)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory for CSV output (default: ./data)")
    parser.add_argument("--dry-run", action="store_true", help="Print stats only, do NOT write files")
    args = parser.parse_args()

    random.seed(args.seed)

    logger.info("Generating %d runs over %d days (seed=%d) …", args.runs, args.days, args.seed)
    run_rows, step_rows = generate_run_batch(args.runs, args.days)
    logger.info("Generation complete: %d runs, %d steps", len(run_rows), len(step_rows))

    if args.dry_run:
        _print_dry_run_summary(run_rows, step_rows)
        logger.info("Dry-run mode – nothing written.")
        return

    _insert_to_csv(run_rows, step_rows, data_dir=args.data_dir)
    _print_dry_run_summary(run_rows, step_rows)


if __name__ == "__main__":
    main()
