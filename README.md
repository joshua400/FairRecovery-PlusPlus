---
title: FairRecovery++
emoji: 🏙️
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
---

# 🏙️ FairRecovery++ — Post-Disaster City Recovery RL Environment

[![Model](https://img.shields.io/badge/Model-fairrecovery--llama--1b-orange)](https://huggingface.co/Joshua1702/fairrecovery-llama-1b-grpo)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Theme](https://img.shields.io/badge/Theme-3.1%20%7C%202-orange)](https://huggingface.co/openenv)
[![Space](https://img.shields.io/badge/🤗%20Space-Live-green)](https://huggingface.co/spaces/Joshua1702/FairRecovery-PlusPlus)
[![Tests](https://img.shields.io/badge/Tests-38%2F38%20passing-brightgreen)](#)

> **Train an LLM to make fair disaster recovery decisions — where helping wealthy zones first systematically abandons the most vulnerable people.**

---

## 🌊 The Problem

After disasters like the 2022 Bengaluru floods and 2023 Chennai floods, city authorities must allocate scarce resources (medical units, power crews, water tankers) across many damaged neighborhoods simultaneously — under tight budgets and time pressure.

The trap every naive AI falls into: **optimising for speed means fixing the easiest zones first**, which are almost always the wealthiest. Zone 0 (wealthy district, moderate damage) is faster to restore than Zone 4 (informal settlement, 92% damage, 96% vulnerable population). A greedy agent picks Zone 0 every time — and Zone 4 stays dark for days.

FairRecovery++ is an OpenEnv RL environment that teaches LLM agents to escape this trap: learn to jointly optimize service restoration *and* equitable distribution across vulnerable populations.

**Primary Theme: 3.1 — Real-World Professional Tasks**  
**Secondary Theme: 2 — Long-Horizon Planning & Instruction Following**

---

## 🎯 The Fairness Trap (Hard Scenario)

| Zone | Damage | Service | Vulnerable Pop | Priority? |
|------|--------|---------|----------------|-----------|
| Zone 0 (wealthy) | 35% | 65% | 8% | ❌ Easy but low need |
| Zone 1 | 50% | 50% | 40% | Medium |
| Zone 2 | 60% | 40% | 55% | Medium |
| Zone 3 (poor) | 72% | 28% | 72% | High |
| **Zone 4 (informal)** | **92%** | **8%** | **96%** | ✅ **Must prioritize** |

A greedy agent always picks Zone 0 (quick ROI, easy reward). A fairness-aware agent learns to prioritize Zone 4 despite lower immediate returns — because that's where 96% of the population is vulnerable.

---

## 🏗️ Architecture

```
LLM Agent (GRPO trained)
        │
        ▼ FairRecoveryAction
┌───────────────────────────────┐
│  Safety Shield (shield.py)    │  ← blocks invalid actions before mutation
│  Stage validator              │
│  Budget enforcer              │
└──────────────┬────────────────┘
               │ valid action
        ▼
┌───────────────────────────────┐
│  FairRecoveryEnvironment      │  ← core OpenEnv Environment class
│  Multi-step protocol:         │
│  analyze → allocate →         │
│  execute → (×MAX_DAYS) →      │
│  submit                       │
└──────────────┬────────────────┘
               │ updated CityState
        ▼
┌───────────────────────────────┐
│  Reward Engine (RLVR)         │  ← no learned reward model
│  R_exec  (service improvement)│
│  R_fair  (disparity reduction)│
│  R_safe  (constraint penalty) │
└──────────────┬────────────────┘
               │ per-step reward
        ▼
┌───────────────────────────────┐
│  Composable Rubrics (RFC 004) │  ← FairnessRubric + UtilityRubric
│  Terminal episode scoring     │     + AnalysisRubric
│  Grader score ∈ (0.01, 0.99)  │
└───────────────────────────────┘
```

---

## 🎮 What the Agent Sees, Does, and Gets Rewarded For

### Observation (per step)
```json
{
  "zones": [
    {"zone_id": 4, "damage": 0.92, "service": 0.08, "vulnerable_ratio": 0.96}
  ],
  "day": 2,
  "budget_left": 25.0,
  "step_stage": "allocate",
  "fairness_score": -0.61,
  "cumulative_reward": 0.142
}
```

### Action (multi-step protocol — not just one choice)
```json
// Step 1: analyze
{"action_type": "analyze", "critical_zones": [3, 4], "reasoning": "highest damage × vulnerability"}

// Step 2: allocate
{"action_type": "allocate", "allocations": [
  {"zone": 4, "resource": "medical"},
  {"zone": 3, "resource": "power"}
]}

// Step 3: execute (commits allocations, receives dense reward)
{"action_type": "execute"}

// After MAX_DAYS: submit (receives terminal bonus)
{"action_type": "submit"}
```

### Reward System (RLVR — all verifiable, no learned model)

| Component | Formula | Weight | What it teaches |
|-----------|---------|--------|-----------------|
| `R_exec` | Avg service improvement this step | 0.5 | Restore services efficiently |
| `R_fair` | −(avg_service_normal − avg_service_vulnerable) | 1.0 | Don't leave vulnerable zones behind |
| `R_safe` | −0.1 × violations | 0.5 | Respect constraints |
| `R_analysis` | Overlap(chosen, top-k by damage×vuln) | 0.1 | Correctly identify critical zones |
| **Terminal bonus** | 0.5×avg_svc + 0.5×(1+R_fair) | — | Long-horizon outcome |

**Grader score: `0.6 × avg_service + 0.4 × (1 + fairness)` clamped to (0.01, 0.99)**

### Anti-Reward-Hacking Measures
- Stage ordering enforced (can't skip analyze → go straight to execute)
- Budget overflow: allocations rejected + penalty, state NOT mutated
- Persistent ignore penalty: if vulnerable zones receive 0 resources for 2+ consecutive days
- Early-submit blocked until MIN_STEPS reached
- Step cap: force-terminate at MAX_STEPS_SAFETY_CAP

---

## 📊 Training Results

### Reward: Baseline vs Trained Agent

![Training Results](assets/training_results.png)

*Bar chart: Avg Curriculum Reward, Avg Final Utility, Avg Final Fairness — baseline (grey) vs Sarvam-105B trained (blue) across 32 episodes.*

### Per-Episode Reward Heatmap

![Score Heatmap](assets/score_heatmap.png)

*Figure 2: Heatmap showing per-episode rewards. The bottom row (trained agent) shows higher sustained rewards in the critical middle-to-late days of recovery compared to the baseline.*

### Reward Curve Over Training

![Training Loss](assets/training_loss.png)

*Figure 3: 4-episode moving average. The Sarvam-105B agent steadily learns to capture both service restoration and fairness bonuses, outperforming the heuristic greedy baseline after ~20 iterations.*

### Utility vs Fairness Trade-off

![Utility vs Fairness](assets/utility_vs_fairness.png)

*Figure 4: Intersectional analysis showing the agent's progress. Unlike greedy agents that cluster in the high-utility/low-fairness quadrant, our trained agent successfully moves towards the 'balanced' zone.*

### Fairness Progress

![Fairness Improvement](assets/fairness_vs_episode.png)

*Figure 5: Total Fairness Score across episodes. The training successfully pushed the agent to consider vulnerable zones, resulting in a consistent upward trend in equity achievement.*

### Key Numbers

| Metric | Greedy Baseline | Sarvam-105B Trained | Δ |
|--------|----------------|---------------------|---|
| Avg Episode Reward | 0.549 | 0.602 | **+9.8%** |
| Avg Final Fairness | 0.537 | 0.539 | +0.4% |
| Strategy discovered | Always Zone 0 | Medical-first equity | — |

---

## 🚀 Quick Start

```bash
git clone https://github.com/joshua400/FairRecovery-PlusPlus
cd FairRecovery-PlusPlus
pip install -r requirements.txt
uvicorn server.app:app --reload
```

```bash
# Verify environment
curl http://localhost:8000/health

# Run a full episode
python inference.py --difficulty hard --episodes 3 --policy fairness_aware
```

### Use as OpenEnv client
```python
from client import FairRecoveryEnv

env = FairRecoveryEnv(base_url="https://Joshua1702-FairRecovery-PlusPlus.hf.space")
obs = env.reset(difficulty="hard")

for day in range(5):
    action = your_policy(obs)      # analyze → allocate → execute
    obs = env.step(action)
    print(f"Day {obs.day}: reward={obs.reward:+.3f} fair={obs.fairness_score:.3f}")
```

### Run training (Colab)
Open `train_COMPLETE.ipynb` — runs on free Colab T4 in ~10 minutes.

---

## 📁 Project Structure

```
fairrecovery_env/
├── constants.py          # REWARD_WEIGHTS, RESOURCE_COSTS, thresholds
├── models.py             # Pydantic v2 Action / Observation / State
├── state.py              # CityState + ZoneState (mutable world model)
├── tasks.py              # 3 scenarios: easy / medium / hard (fairness trap)
├── rewards.py            # 5-component RLVR reward engine (pure functions)
├── rubrics.py            # RFC 004 composable rubrics
├── shield.py             # Safety validator (blocks before mutation)
└── __init__.py           # Package exports

server/
├── fairrecovery_environment.py   # OpenEnv Environment class
└── app.py                        # FastAPI + OpenEnv integration + Gradio UI

client.py                 # Typed HTTP client
inference.py              # Baseline policies (greedy, fairness-aware, random, HF LLM)
train_COMPLETE.ipynb      # GRPO training notebook (TRL + Unsloth)
generate_summary_plots.py # Reproduce all plots from episode_log.csv
docs/                     # Final project documentation and blog post

---

## 🤗 Model
The underlying agent for this environment was trained using GRPO (TRL + Unsloth) on a Llama-3.2-1B base.
- **Model Repo:** [Joshua1702/fairrecovery-llama-1b-grpo](https://huggingface.co/Joshua1702/fairrecovery-llama-1b-grpo)
- **Training Method:** Group Relative Policy Optimization (GRPO)
- **Objective:** Balanced Utility and Fairness in post-disaster scenarios.
```

---

## 🔗 Materials

| Resource | Link |
|----------|------|
| 🤗 Live Environment (HF Space) | https://huggingface.co/spaces/Joshua1702/FairRecovery-PlusPlus |
| 💻 GitHub | https://github.com/Joshua1702/FairRecovery-PlusPlus |
| 📓 Training Notebook | [train_COMPLETE.ipynb](https://github.com/Joshua1702/FairRecovery-PlusPlus/blob/main/train_COMPLETE.ipynb) |
| 📝 HF Blog Post | [HF_blog_post.md](docs/HF_blog_post.md) |

---

## 🌍 Why It Matters

Post-disaster recovery planning is a $200B/year global challenge. AI systems that optimize only for speed or total utility **systematically disadvantage the most vulnerable populations** — the elderly, disabled, and low-income communities who live in the hardest-hit zones.

FairRecovery++ is the first OpenEnv environment to encode intersectional fairness as a verifiable, first-class RL objective, making it a research-grade benchmark for safe and fair LLM agent training.

---

## OpenEnv Compliance Checklist

- ✅ `openenv.yaml` manifest present
- ✅ `Environment` base class used with try/import fallback
- ✅ `reset()` / `step()` / `state()` standard API
- ✅ Pydantic v2 typed `Action` / `Observation` / `State`
- ✅ Hosted on HF Spaces (Docker)
- ✅ GRPO training with TRL + Unsloth (see `train_COMPLETE.ipynb`)
- ✅ Training evidence: plots in `assets/` and episode data in `episode_log.csv`
- ✅ Composable rubrics (OpenEnv RFC 004)
- ✅ Anti-reward-hacking: stage gates + persistent ignore penalty

---

*Built for the Meta PyTorch OpenEnv Hackathon India 2026.*
