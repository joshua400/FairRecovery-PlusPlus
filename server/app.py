"""
FairRecovery++ — FastAPI Application.

Updated with Live LLM (Hugging Face) support and Training Data logging.
"""

from __future__ import annotations
import os, sys
import re
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import RedirectResponse

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fairrecovery_env.models import FairRecoveryAction
from server.fairrecovery_environment import FairRecoveryEnvironment
from inference import greedy_policy, fairness_aware_policy, HFInferencePolicy, TrainingLogger

def _build_app():
    import gradio as gr
    app = FastAPI(title="FairRecovery++ RL Environment", version="2.0.0")
    _env = FairRecoveryEnvironment()
    _logger = TrainingLogger()

    @app.post("/reset")
    async def reset(difficulty: str = "medium", episode_id: Optional[str] = None):
        return _env.reset(difficulty=difficulty, episode_id=episode_id).model_dump()

    @app.post("/step")
    async def step(request: Request):
        payload = await request.json()
        action = FairRecoveryAction(**payload)
        return _env.step(action).model_dump()

    def run_simulation(policy_type: str, hf_token: str):
        env = FairRecoveryEnvironment()
        obs = env.reset(task_id="multi_disaster_hard")
        
        logs = []
        logs.append(f"### 🚀 LIVE SESSION: {policy_type.upper()}")
        
        # Policy Selection
        if policy_type == "Live LLM (Llama-3)":
            if not hf_token:
                return "### ❌ Error\nPlease provide a Hugging Face Token to use the Live LLM.", ""
            policy_fn = HFInferencePolicy(token=hf_token)
        elif policy_type == "Baseline (Greedy)":
            policy_fn = greedy_policy
        else:
            policy_fn = fairness_aware_policy
        
        done = False
        while not done:
            action = policy_fn(obs)
            prev_obs = obs # For logging
            obs = env.step(action)
            
            # Log for training
            _logger.log_step(prev_obs, action, obs.reward)
            
            if action.action_type == "execute":
                logs.append(f"✅ **Day {obs.day-1} Complete**")
                z4 = obs.zones[4]
                logs.append(f"  - Zone 4 Status: Damage {z4.damage:.2f}, Equity Index: {obs.fairness_score:.2f}")
                if action.reasoning:
                    logs.append(f"  - *AI Reasoning:* {action.reasoning}")
            
            done = obs.done

        res_eval = "🟢 **EQUITY ACHIEVED**" if obs.fairness_score > 0.8 else "🔴 **NEGLECT DETECTED**"
        result_text = f"### 🏆 FINAL OUTCOME\n- **Reward Score:** {obs.cumulative_reward:.3f}\n- **Equity Index:** {obs.fairness_score:.3f}\n\n{res_eval}\n\n*Trajectory saved to training_data/ for RL refinement.*"
        return "\n".join(logs), result_text

    with gr.Blocks(title="FairRecovery++", theme=gr.themes.Soft()) as gradio_app:
        gr.Markdown("# 🏗️ FairRecovery++: Real-time LLM Simulator")
        with gr.Tabs():
            with gr.Tab("Simulation"):
                with gr.Row():
                    policy = gr.Dropdown(
                        choices=["Baseline (Greedy)", "Fairness Aware (Heuristic)", "Live LLM (Llama-3)"], 
                        value="Baseline (Greedy)",
                        label="Agent Strategy"
                    )
                    hf_token = gr.Textbox(label="Hugging Face Token (optional for Live LLM)", type="password")
                
                btn = gr.Button("Run Live Simulation", variant="primary")
                logs = gr.Markdown()
                results = gr.Markdown()
                btn.click(run_simulation, inputs=[policy, hf_token], outputs=[logs, results])
            with gr.Tab("README"):
                readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md")
                with open(readme_path, "r", encoding="utf-8") as f:
                    gr.Markdown(f.read())

    @app.get("/")
    async def root(): return RedirectResponse(url="/ui/")
    return gr.mount_gradio_app(app, gradio_app, path="/ui")

app = _build_app()
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
