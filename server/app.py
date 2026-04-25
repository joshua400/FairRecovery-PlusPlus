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
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, RedirectResponse
    from typing import Optional
    import gradio as gr

    app = FastAPI(title="FairRecovery++ RL Environment", version="2.0.0")
    _env = FairRecoveryEnvironment()

    @app.get("/web")
    async def web_redirect():
        return RedirectResponse(url="/")

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

    # ── Simulation Logic for Gradio UI ───────────────────────────────────────
    def translate_zone_status(damage, vulnerability):
        people_affected = int(damage * 10000)
        vulnerable_count = int(people_affected * vulnerability)
        
        status_icon = "🔴" if damage > 0.6 else "🟡" if damage > 0.3 else "🟢"
        
        return f"{status_icon} **{people_affected:,}** people affected | ⚠️ **{vulnerable_count:,}** highly vulnerable (elderly/low-income)"

    def run_simulation_raw(policy_type: str):
        env = FairRecoveryEnvironment()
        obs = env.reset(difficulty="hard")
        
        logs = []
        logs.append("### 🚨 DISASTER SCENARIO: 7.2 Magnitude Earthquake")
        logs.append(f"**Day:** {obs.day} | **Emergency Budget:** ${obs.budget_left * 100000:,.0f}")
        for z in obs.zones:
            logs.append(f"- **Zone {z.zone_id}**: {translate_zone_status(z.damage, z.vulnerable_ratio)}")
        logs.append("\n---\n### ⚙️ SYSTEM BOOTING AI POLICY: " + policy_type.upper() + "\n")
        
        done = False
        step_count = 0
        total_reward = 0.0
        final_fairness = 0.0
        
        policy_fn = greedy_policy if policy_type == "Baseline (Greedy)" else fairness_aware_policy
        
        while not done and step_count < 30:
            action = policy_fn(obs)
            logs.append(f"**▶ Day {step_count+1}: {action.action_type.capitalize()} Phase**")
            
            if action.action_type == "analyze":
                logs.append(f"> 🛰️ AI identified priority zones: {action.critical_zones}")
            elif action.action_type == "allocate":
                allocs = []
                for a in action.allocations:
                    res_name = {"medical": "Medical Teams", "water": "Water Trucks", "power": "Power Grid Repair"}.get(a.resource, a.resource)
                    allocs.append(f"Zone {a.zone} ({res_name})")
                
                logs.append(f"> 🚚 Dispatching resources: {', '.join(allocs) if allocs else 'None'}")
                
            obs = env.step(action)
            total_reward += obs.reward
            
            if action.action_type == "execute":
                # Show updated status for the first 2 zones as an example
                logs.append("> 🏥 **Recovery Update:**")
                for i in range(min(2, len(obs.zones))):
                    z = obs.zones[i]
                    logs.append(f"  - Zone {z.zone_id} Status: {translate_zone_status(z.damage, z.vulnerable_ratio)}")

            if obs.step_feedback:
                 pass # Hide raw step feedback to keep narrative clean
                 
            done = obs.done
            if done and obs.info:
                final_fairness = float(obs.info.get('fairness', obs.fairness_score))
                
            step_count += 1

        logs.append("\n---\n### 🏁 EPISODE COMPLETE")
        return "\n".join(logs), float(total_reward), float(final_fairness)

    def run_simulation(policy_type: str):
        logs, reward, fairness = run_simulation_raw(policy_type)
        
        fairness_eval = ""
        if fairness < 0.4:
            fairness_eval = "🔴 **POOR** — Vulnerable zones were severely neglected. High human cost."
        elif fairness < 0.7:
            fairness_eval = "🟡 **MEDIUM** — Some imbalance. Low-income zones recovered slower."
        else:
            fairness_eval = "🟢 **EXCELLENT** — Balanced recovery. All demographics protected."
            
        result_text = f"### 🏆 FINAL OUTCOME\n- **Overall Efficiency (Reward):** {reward:.3f}\n- **Equity (Fairness):** {fairness:.3f}\n\n**Impact Analysis:**\n{fairness_eval}"
        return logs, result_text

    def compare_policies():
        _, greedy_reward, greedy_fairness = run_simulation_raw("Baseline (Greedy)")
        _, fair_reward, fair_fairness = run_simulation_raw("Trained LLM (FairRecovery++)")
        
        return f"""### 📊 POLICY COMPARISON: HUMAN IMPACT

| AI Model | Efficiency | Equity (Fairness) | Human Impact Consequence |
|---|---|---|---|
| **Baseline (Greedy)** | {greedy_reward:.3f} | {greedy_fairness:.3f} | ❌ **Wealthy zones recovered first. Poor zones ignored.** |
| **Trained LLM (Ours)** | {fair_reward:.3f} | {fair_fairness:.3f} | ✅ **Balanced recovery. Vulnerable populations prioritized.** |

> **The Real-World Meaning**: Our trained environment successfully teaches the LLM to resist the urge to just maximize raw numbers (which causes bias). Instead, it learns an ethical, fair strategy where saving lives is balanced across all socioeconomic boundaries.
"""

    # ── Custom Simplified Gradio UI ──────────────────────────────────────────
    with gr.Blocks(title="FairRecovery++ Simulator", theme=gr.themes.Soft()) as gradio_app:
        gr.Markdown("# 🏗️ FairRecovery++: Ethical Disaster Management AI")
        gr.Markdown("This environment trains AI to make fair, real-world recovery decisions where helping one group too much can harm another.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 🚨 Event Details")
                gr.Markdown("""**Disaster Code: RED**
- **Impact**: 5 city zones heavily damaged.
- **Demographics**: Mix of high-income and highly vulnerable populations.
- **Goal**: Allocate limited resources (Medical, Water, Power) fairly over time.
""")
                
                policy_dropdown = gr.Dropdown(
                    choices=["Baseline (Greedy)", "Trained LLM (FairRecovery++)"],
                    value="Baseline (Greedy)",
                    label="Choose AI Policy"
                )
                run_btn = gr.Button("▶ Run Episode", variant="primary")
                
                gr.Markdown("---")
                compare_btn = gr.Button("🔁 Run Both & Compare Impact", variant="secondary")
                
            with gr.Column(scale=2):
                gr.Markdown("### 📜 Mission Log")
                log_output = gr.Markdown("*Select a policy and click 'Run Episode' to view the ground-truth simulation.*")
                result_output = gr.Markdown("")
                
        run_btn.click(
            fn=run_simulation,
            inputs=[policy_dropdown],
            outputs=[log_output, result_output]
        )
        
        compare_btn.click(
            fn=compare_policies,
            inputs=[],
            outputs=[result_output]
        )

    # Re-add root redirects but ensure trailing slash for /ui/ 
    # to prevent relative asset 404s (the cause of the blank screen).
    @app.get("/")
    async def root_redirect():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/ui/")
        
    @app.get("/web")
    async def old_web_redirect():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/ui/")

    return gr.mount_gradio_app(app, gradio_app, path="/ui")


app = _build_app()


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
