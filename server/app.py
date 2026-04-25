"""
FairRecovery++ — FastAPI Application.

Mirrors hallucination-detector-gym server/app.py pattern exactly.
"""

from __future__ import annotations
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from openenv.core.env_server.http_server import create_app
    _OPENENV_AVAILABLE = True
except ImportError:
    _OPENENV_AVAILABLE = False

from fairrecovery_env.models import FairRecoveryAction, FairRecoveryObservation
from fairrecovery_env.logging_config import configure_logging
from server.fairrecovery_environment import FairRecoveryEnvironment

configure_logging(json_output=True, log_level="INFO")


def _build_app():
    if _OPENENV_AVAILABLE:
        return create_app(FairRecoveryEnvironment, FairRecoveryAction,
                          FairRecoveryObservation, env_name="fairrecovery",
                          max_concurrent_envs=1)

    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from typing import Optional

    app = FastAPI(title="FairRecovery++ RL Environment", version="2.0.0",
                  description="Adaptive multi-agent post-disaster recovery RL environment.")
    _env = FairRecoveryEnvironment()

    @app.post("/reset", response_model=None)
    async def reset(difficulty: str = "medium", episode_id: Optional[str] = None):
        return _env.reset(difficulty=difficulty, episode_id=episode_id).model_dump()

    @app.post("/step", response_model=None)
    async def step(request: Request):
        try:
            payload = await request.json()
            action = FairRecoveryAction(**payload)
        except Exception as exc:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid action payload: {exc}"},
            )
        try:
            return _env.step(action).model_dump()
        except Exception as exc:
            return JSONResponse(
                status_code=400,
                content={"error": f"step failed: {exc}"},
            )

    @app.get("/state")
    async def state():
        return _env.state.model_dump()

    @app.get("/schema")
    async def schema():
        return {"action": FairRecoveryAction.model_json_schema(),
                "observation": FairRecoveryObservation.model_json_schema()}

    @app.get("/health")
    async def health():
        return {"status": "ok", "env": "FairRecovery++", "version": "2.0.0"}

    return app


app = _build_app()


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
