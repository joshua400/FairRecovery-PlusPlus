"""
FairRecovery — FastAPI Application.

Mirrors hallucination-detector-gym server/app.py pattern exactly.
Provides: POST /reset, POST /step, GET /state, GET /schema, GET /health.

Usage:
    # Development:
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1

    # Via OpenEnv CLI:
    uv run --project . server
"""

from __future__ import annotations

import os
import sys

# Ensure fairrecovery_env is importable from the server
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from openenv.core.env_server.http_server import create_app, create_fastapi_app
    _OPENENV_AVAILABLE = True
except ImportError:
    _OPENENV_AVAILABLE = False

from fairrecovery_env.models import FairRecoveryAction, FairRecoveryObservation
from fairrecovery_env.logging_config import configure_logging
from server.fairrecovery_environment import FairRecoveryEnvironment

# Configure structured JSON logging
configure_logging(json_output=True, log_level="INFO")


def _build_app():
    """Build the FastAPI application with graceful fallback."""

    if _OPENENV_AVAILABLE:
        return create_app(
            FairRecoveryEnvironment,
            FairRecoveryAction,
            FairRecoveryObservation,
            env_name="fairrecovery",
            max_concurrent_envs=1,
        )

    # ── Fallback: manual FastAPI app (runs without openenv installed) ─────────
    from fastapi import FastAPI, HTTPException, Body
    from typing import Any, Dict, Optional

    app = FastAPI(
        title="FairRecovery RL Environment",
        version="1.0.0",
        description=(
            "Post-disaster city recovery RL environment. "
            "Endpoints: POST /reset, POST /step, GET /state, GET /health."
        ),
    )

    _env = FairRecoveryEnvironment()

    @app.post("/reset", response_model=None)
    async def reset(
        difficulty: str = "medium",
        episode_id: Optional[str] = None,
    ):
        """Reset the environment and return initial observation."""
        obs = _env.reset(difficulty=difficulty, episode_id=episode_id)
        return obs.model_dump()

    @app.post("/step", response_model=None)
    async def step(action: FairRecoveryAction):
        """Execute one agent action."""
        obs = _env.step(action)
        return obs.model_dump()

    @app.get("/state")
    async def state():
        """Return current internal environment state."""
        return _env.state.model_dump()

    @app.get("/schema")
    async def schema():
        """Return action and observation JSON schemas."""
        return {
            "action":      FairRecoveryAction.model_json_schema(),
            "observation": FairRecoveryObservation.model_json_schema(),
        }

    @app.get("/health")
    async def health():
        """Health check."""
        return {"status": "ok", "env": "FairRecovery", "version": "1.0.0"}

    return app


app = _build_app()


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point for `uv run --project . server`."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
