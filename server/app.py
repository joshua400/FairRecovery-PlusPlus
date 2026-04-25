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
    # ── Clean README.md on disk for OpenEnv compatibility ───────────────────
    # OpenEnv core looks for README.md in the root. If it has a HF YAML header,
    # the parser might fail. We clean it in-place in the container.
    readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md")
    if os.path.exists(readme_path):
        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    cleaned = parts[2].strip()
                    with open(readme_path, "w", encoding="utf-8") as f:
                        f.write(cleaned)
        except Exception as e:
            print(f"Warning: Failed to clean README.md: {e}")

    if not _OPENENV_AVAILABLE:
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
        from typing import Optional

        app = FastAPI(title="FairRecovery++ RL Environment", version="2.0.0")
        _env = FairRecoveryEnvironment()

        @app.post("/reset")
        async def reset(difficulty: str = "medium", episode_id: Optional[str] = None):
            return _env.reset(difficulty=difficulty, episode_id=episode_id).model_dump()

        @app.post("/step")
        async def step(request: Request):
            payload = await request.json()
            action = FairRecoveryAction(**payload)
            return _env.step(action).model_dump()

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        return app

    # Use the high-level create_app which handles everything correctly
    # now that README.md is cleaned on disk.
    return create_app(
        FairRecoveryEnvironment,
        FairRecoveryAction,
        FairRecoveryObservation,
        env_name="fairrecovery",
        max_concurrent_envs=1
    )


app = _build_app()


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
