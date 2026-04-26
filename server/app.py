"""
FairRecovery++ — FastAPI Application.

Fixed Gradio input mismatch and layout.
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
from inference import greedy_policy, fairness_aware_policy, HFInferencePolicy, TrainingLogger, TrainedInferencePolicy

def _build_app():
    import gradio as gr
    from fastapi.staticfiles import StaticFiles
    
    app = FastAPI(title="FairRecovery++ RL Environment", version="2.0.0")
    
    # Static files for assets
    assets_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
    if os.path.exists(assets_path):
        app.mount("/assets", StaticFiles(directory=assets_path), name="assets")

    _env = FairRecoveryEnvironment()
    _logger = TrainingLogger()
    _trained_llama = None
    _trained_qwen = None

    @app.get("/health")
    async def health(): return {"status": "healthy"}

    @app.post("/reset")
    async def reset(difficulty: str = "medium", episode_id: Optional[str] = None):
        return _env.reset(difficulty=difficulty, episode_id=episode_id).model_dump()

    @app.post("/step")
    async def step(request: Request):
        payload = await request.json()
        action = FairRecoveryAction(**payload)
        return _env.step(action).model_dump()

    def run_simulation(policy_type: str, hf_token: str = ""):
        nonlocal _trained_llama, _trained_qwen
        env = FairRecoveryEnvironment()
        obs = env.reset(task_id="multi_disaster_hard")
        
        logs = []
        logs.append(f"### 🚀 SESSION: {policy_type.upper()}")
        
        # Policy Selection
        try:
            if policy_type == "Live LLM (Llama-3)":
                if not hf_token or hf_token.strip() == "":
                    return "### ❌ Error\nPlease provide a Hugging Face Token in the sidebar to use the Live LLM.", ""
                policy_fn = HFInferencePolicy(token=hf_token)
            elif "Qwen" in policy_type:
                if _trained_qwen is None:
                    logs.append("⏳ *Loading Trained Qwen-7B model into GPU...*")
                    _trained_qwen = TrainedInferencePolicy(model_name="Joshua1702/fairrecovery-Qwen2.5-7B-GRPO")
                policy_fn = _trained_qwen
            elif "Llama-1B-GRPO" in policy_type:
                if _trained_llama is None:
                    logs.append("⏳ *Loading Trained Llama-1B model into GPU...*")
                    _trained_llama = TrainedInferencePolicy(model_name="Joshua1702/fairrecovery-llama-1b-grpo")
                policy_fn = _trained_llama
            elif policy_type == "Baseline (Greedy)":
                policy_fn = greedy_policy
            else:
                policy_fn = fairness_aware_policy
        except Exception as e:
            logs.append(f"❌ **Model Error:** {str(e)}")
            return "\n".join(logs), f"### ❌ Initialization Failed\nCheck logs for details."
        
        done = False
        while not done:
            action = policy_fn(obs)
            prev_obs = obs
            obs = env.step(action)
            _logger.log_step(prev_obs, action, obs.reward)
            
            if action.action_type == "execute":
                logs.append(f"✅ **Day {obs.day-1} Complete** (Equity: {obs.fairness_score:.2f})")
            
            done = obs.done

        res_eval = "🟢 **EQUITY ACHIEVED**" if obs.fairness_score > 0.8 else "🔴 **NEGLECT DETECTED**"
        result_text = f"### 🏆 FINAL OUTCOME\n- **Reward:** {obs.cumulative_reward:.3f}\n- **Equity Index:** {obs.fairness_score:.3f}\n\n{res_eval}"
        return "\n".join(logs), result_text

    with gr.Blocks(title="FairRecovery++", theme=gr.themes.Soft()) as gradio_app:
        gr.Markdown("# 🏗️ FairRecovery++: Disaster Response Simulator")
        
        with gr.Row():
            with gr.Column(scale=1):
                policy = gr.Dropdown(
                    choices=[
                        "Baseline (Greedy)", 
                        "Fairness Aware (Heuristic)", 
                        "Live LLM (Llama-3)", 
                        "Trained Model (Llama-1B-GRPO)", 
                        "Trained Model (Qwen-2.5-7B-GRPO)"
                    ], 
                    value="Trained Model (Qwen-2.5-7B-GRPO)",
                    label="Agent Strategy"
                )
                token_input = gr.Textbox(label="Hugging Face Token", placeholder="Enter token for Live LLM...", type="password")
                btn = gr.Button("Run Simulation", variant="primary")
            
            with gr.Column(scale=2):
                results = gr.Markdown("### Results will appear here...")
                logs = gr.Markdown("### Simulation Logs")

        btn.click(fn=run_simulation, inputs=[policy, token_input], outputs=[logs, results])

        with gr.Tab("Analysis & Fairness"):
            gr.Markdown("## 📊 Training Results & Fairness Trends")
            
            # Use absolute filesystem paths for gr.Image
            base_dir = os.path.dirname(os.path.dirname(__file__))
            asset_plots = os.path.join(base_dir, "evidence", "plots")
            results_img = os.path.join(asset_plots, "training_results.png")
            heatmap_img = os.path.join(asset_plots, "score_heatmap.png")
            loss_img = os.path.join(asset_plots, "training_loss.png")
            fair_img = os.path.join(asset_plots, "fairness_vs_episode.png")
            comp_img = os.path.join(asset_plots, "component_rewards.png")
            model_comp_img = os.path.join(asset_plots, "model_comparison.png")

            with gr.Row():
                gr.Image(model_comp_img if os.path.exists(model_comp_img) else None, label="Model Comparison (Llama vs Qwen)")
            with gr.Row():
                gr.Image(results_img if os.path.exists(results_img) else None, label="Trained vs Baseline")
                gr.Image(heatmap_img if os.path.exists(heatmap_img) else None, label="Episode Rewards")
            with gr.Row():
                gr.Image(loss_img if os.path.exists(loss_img) else None, label="Reward Convergence")
                gr.Image(fair_img if os.path.exists(fair_img) else None, label="Fairness Improvement")
            with gr.Row():
                gr.Image(comp_img if os.path.exists(comp_img) else None, label="Component Breakdown (Utility vs Fairness)")

        with gr.Tab("README"):
            readme_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md")
            if os.path.exists(readme_path):
                with open(readme_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    content = re.sub(r'^---.*?---', '', content, flags=re.DOTALL)
                    # Correctly resolve images for Gradio Markdown
                    content = content.replace("evidence/plots/", "file/evidence/plots/")
                    gr.Markdown(content)

    @app.get("/")
    async def root(): return RedirectResponse(url="/ui/")
    return gr.mount_gradio_app(app, gradio_app, path="/ui")

app = _build_app()
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
