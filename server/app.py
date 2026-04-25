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

    from openenv.core.env_server.http_server import create_fastapi_app
    from openenv.core.env_server.web_interface import (
        WebInterfaceManager,
        load_environment_metadata,
        get_quick_start_markdown,
    )
    from openenv.core.env_server.gradio_ui import build_gradio_app
    from openenv.core.env_server.gradio_theme import OPENENV_GRADIO_THEME, OPENENV_GRADIO_CSS
    import gradio as gr

    # 1. Base FastAPI app
    fastapi_app = create_fastapi_app(
        FairRecoveryEnvironment,
        FairRecoveryAction,
        FairRecoveryObservation,
        max_concurrent_envs=1,
    )

    # 2. Metadata & Manager
    metadata = load_environment_metadata(FairRecoveryEnvironment, "fairrecovery")
    
    # 3. Manual README loading to ensure it's not empty
    readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md")
    readme_content = ""
    if os.path.exists(readme_path):
        with open(readme_path, "r", encoding="utf-8") as f:
            readme_content = f.read()
            # Strip YAML header if present for the web UI
            if readme_content.startswith("---"):
                parts = readme_content.split("---", 2)
                if len(parts) >= 3:
                    readme_content = parts[2].strip()

    web_manager = WebInterfaceManager(
        FairRecoveryEnvironment,
        FairRecoveryAction,
        FairRecoveryObservation,
        metadata,
    )

    # 4. Build Gradio app
    gradio_blocks = build_gradio_app(
        web_manager,
        FairRecoveryAction,
        FairRecoveryObservation,
        metadata,
        title="FairRecovery++ Playground",
        quick_start_md=readme_content,
    )

    # 5. Mount
    return gr.mount_gradio_app(
        fastapi_app,
        gradio_blocks,
        path="/web",
        theme=OPENENV_GRADIO_THEME,
        css=OPENENV_GRADIO_CSS,
    )


app = _build_app()


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
