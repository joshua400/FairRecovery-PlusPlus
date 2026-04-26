"""
FairRecovery++ — FastAPI Application.

Perfectly aligned with the refactored environment and reference structure.
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

def llm_policy(obs: FairRecoveryObservation):
    """Real-time LLM inference using HF API."""
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        return fairness_aware_policy(obs)
        
    try:
        zones_str = '\n'.join([f"Zone {z.zone_id}: damage={z.damage:.2f}, vulnerable={z.vulnerable_ratio:.2f}" for z in obs.zones])
        prompt = f"System: You are an AI allocating resources fairly. Respond with JSON action.\nUser: Day {obs.day}. Budget {obs.budget_left}. Zones:\n{zones_str}\nWhat is your next action?"
        
        response = requests.post(
            "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-3B-Instruct",
            headers={"Authorization": f"Bearer {hf_token}"},
            json={"inputs": prompt, "parameters": {"max_new_tokens": 100, "temperature": 0.1}},
            timeout=5
        )
        if response.status_code == 200:
            text = response.json()[0]["generated_text"]
            match = re.search(r'\{.*?\}', text, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return FairRecoveryAction(**data)
    except Exception as e:
        print(f"LLM API failed: {e}")
        
    return fairness_aware_policy(obs)

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

    # ── Simulation Logic for Gradio UI ───────────────────────────────────────
    def translate_zone_status(damage, vulnerability):
        people_affected = int(damage * 10000)
        vulnerable_count = int(people_affected * vulnerability)
        status_icon = "🔴" if damage > 0.6 else "🟡" if damage > 0.3 else "🟢"
        return f"{status_icon} **{people_affected:,}** people | ⚠️ **{vulnerable_count:,}** vulnerable"

    def run_simulation(policy_type: str):
        env = FairRecoveryEnvironment()
        obs = env.reset(task_id="multi_disaster_hard")
        
        logs = []
        logs.append(f"### 🚨 SCENARIO: {policy_type.upper()}")
        
        done = False
        step_count = 0
        policy_fn = greedy_policy if policy_type == "Baseline (Greedy)" else llm_policy
        
        while not done and step_count < 30:
            action = policy_fn(obs)
            obs = env.step(action)
            
            logs.append(f"**Day {obs.day}**: AI performed {action.action_type.value}")
            if action.action_type == "allocate":
                for a in (action.allocations or []):
                    logs.append(f"  - Dispatched {a.resource.value} to Zone {a.zone}")
            
            done = obs.done
            step_count += 1

        res_eval = "🟢 **EQUITY ACHIEVED**" if obs.fairness_score > 0.8 else "🔴 **NEGLECT DETECTED**"
        result_text = f"### 🏆 FINAL OUTCOME\n- **Reward:** {obs.cumulative_reward:.3f}\n- **Equity:** {obs.fairness_score:.3f}\n\n{res_eval}"
        return "\n".join(logs), result_text

    with gr.Blocks(title="FairRecovery++", theme=gr.themes.Soft()) as gradio_app:
        gr.Markdown("# 🏗️ FairRecovery++")
        with gr.Tabs():
            with gr.Tab("Simulation"):
                policy = gr.Dropdown(choices=["Baseline (Greedy)", "Trained LLM"], value="Baseline (Greedy)")
                btn = gr.Button("Run Simulation")
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
