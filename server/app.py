"""
FairRecovery++ — FastAPI Application.

Updated simulation loop for phase-aware progression.
"""

from __future__ import annotations
import os, sys
import re
import requests
import json
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fairrecovery_env.models import FairRecoveryAction, FairRecoveryObservation
from server.fairrecovery_environment import FairRecoveryEnvironment
from inference import greedy_policy, fairness_aware_policy

def _build_app():
    import gradio as gr
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

    def run_simulation(policy_type: str):
        env = FairRecoveryEnvironment()
        obs = env.reset(task_id="multi_disaster_hard")
        
        logs = []
        logs.append(f"### 🚨 SCENARIO: {policy_type.upper()}")
        
        done = False
        policy_fn = greedy_policy if policy_type == "Baseline (Greedy)" else fairness_aware_policy
        
        while not done:
            action = policy_fn(obs)
            obs = env.step(action)
            
            if action.action_type == "execute":
                logs.append(f"✅ **Day {obs.day-1} Complete**")
                # Show status of critical zones
                z4 = obs.zones[4]
                logs.append(f"  - Zone 4 (Vulnerable) Status: Damage {z4.damage:.2f}, Svc {z4.service_level:.2f}")
            
            done = obs.done

        res_eval = "🟢 **EQUITY ACHIEVED**" if obs.fairness_score > 0.8 else "🔴 **NEGLECT DETECTED**"
        result_text = f"### 🏆 FINAL OUTCOME\n- **Reward Score:** {obs.cumulative_reward:.3f}\n- **Equity Index:** {obs.fairness_score:.3f}\n\n{res_eval}"
        return "\n".join(logs), result_text

    with gr.Blocks(title="FairRecovery++", theme=gr.themes.Soft()) as gradio_app:
        gr.Markdown("# 🏗️ FairRecovery++: Disaster Response Simulator")
        with gr.Tabs():
            with gr.Tab("Simulation"):
                policy = gr.Dropdown(choices=["Baseline (Greedy)", "Trained LLM (Fairness Aware)"], value="Baseline (Greedy)")
                btn = gr.Button("Run Simulation", variant="primary")
                logs = gr.Markdown()
                results = gr.Markdown()
                btn.click(run_simulation, inputs=[policy], outputs=[logs, results])
            with gr.Tab("README"):
                readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md")
                with open(readme_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content.startswith("---"):
                        content = re.sub(r"^---.*?---", "", content, flags=re.DOTALL)
                    gr.Markdown(content)

    @app.get("/")
    async def root(): return RedirectResponse(url="/ui/")
    return gr.mount_gradio_app(app, gradio_app, path="/ui")

app = _build_app()
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
