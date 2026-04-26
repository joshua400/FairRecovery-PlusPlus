---
title: FairRecovery++
emoji: 🏙️
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
---

# 🏙️ FairRecovery++: Teaching LLMs to Make Fair Disaster Recovery Decisions

🚨 Problem: AI systems optimize speed — but silently ignore the most vulnerable.

💡 Solution: FairRecovery++ trains LLMs to make recovery decisions that are both efficient AND fair.

[![Llama-1B](https://img.shields.io/badge/Model-Llama--3.2--1B-orange)](https://huggingface.co/Joshua1702/fairrecovery-Llama-3.2-1B)
[![Qwen-7B](https://img.shields.io/badge/Model-Qwen--2.5--7B-purple)](https://huggingface.co/Joshua1702/fairrecovery-Qwen2.5-7B-GRPO)
[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://github.com/meta-pytorch/OpenEnv)
[![Theme](https://img.shields.io/badge/Theme-3.1%20%7C%202-orange)](https://huggingface.co/openenv)
[![Space](https://img.shields.io/badge/🤗%20Space-Live-green)](https://huggingface.co/spaces/Joshua1702/FairRecovery-PlusPlus)

## 🚀 Key Results (Compared to Greedy Baseline)

- 📈 Reward improved by **+57%**
- ⚖ Fairness improved by **+24%** (0.73 → 0.91)
- 🧠 Learned to prioritize vulnerable zones

👉 Trained agents outperform greedy baselines in both efficiency and equity.

---
When disaster strikes — floods in Chennai, earthquakes, hurricanes — city authorities face an impossible-looking problem: limited crews, limited budget, dozens of damaged neighborhoods, and no time. Most AI systems trained to help optimize for speed: fix what's easiest first, maximize total service restored.

The problem? **Easy to fix almost always means wealthy.** The informal settlements, the elderly care facilities, and the low-income neighborhoods that took the hardest hit? They wait the longest.

We built **FairRecovery++** to train LLM agents that escape this trap using **GRPO (Group Relative Policy Optimization)**.

---

## What is FairRecovery++?

FairRecovery++ is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv) RL environment where an AI agent acts as a post-disaster Recovery Planner for a simulated city of 5 zones. Each episode spans 10 simulated recovery days. The agent must:

1. **Analyze** which zones are most critically damaged.
2. **Allocate** limited resources (medical units, power crews, water tankers, housing repairs).
3. **Execute** the allocation and observe the outcome.
4. Repeat across multiple days, then **submit** a final recovery plan.

The environment rewards the agent not just for total service restored, but for **equitable** service restored — measured by the gap between how well vulnerable zones are served versus non-vulnerable ones.

---

## 💡 Why This Is New (Not Just Another RL Environment)

Unlike existing disaster RL systems that optimize only efficiency, FairRecovery++ introduces a **“Fairness Trap”** where greedy policies fail.

We are the first to:
- Encode fairness as a verifiable RL objective
- Train LLM planners using GRPO on long-horizon recovery decisions
- Prevent reward hacking with safety-aware constraints

---

## ⚖ Fairness Metric

We define fairness as:

> **Fairness = 1 − service disparity across zones**

Where disparity is the average difference in service levels.

- Fairness = 1 → perfectly equal recovery
- Fairness ↓ → unequal allocation

This makes fairness:
- ✔ hard to game

This ensures fairness cannot be faked — improving fairness requires real redistribution of resources.

---

## 🎯 The Fairness Trap

The hard scenario is deliberately designed with a trap:
- **Zone 0** (wealthy district): 35% damage, easy to fix, 8% vulnerable population.
- **Zone 4** (informal settlement): 92% damage, 96% vulnerable population.

A naive utility-maximizing agent always picks Zone 0: lower cost, faster payoff, higher immediate reward. Zone 4 gets ignored.

Our trained **Qwen-7B** and **Llama-3.2-1B** agents learn to prioritize Zone 4 — because its 96% vulnerable population deserves equitable access to recovery services, even if it costs more per unit of service gained.

---

## 🔍 Before vs After Training

### ❌ Baseline (Greedy Policy)
- Prioritizes low-cost zones
- Ignores high-vulnerability areas
- Leads to unfair recovery

### ✅ Trained Agent (Fair-GRPO-RLVR)
- Identifies critical zones (high damage + vulnerability)
- Allocates resources more equitably
- Improves both fairness and overall recovery

---

## 🤗 Trained Models

We provide two pre-trained models optimized for this environment using GRPO (TRL + Unsloth):

*   **Premium Agent (Qwen-7B)**: [Joshua1702/fairrecovery-Qwen2.5-7B-GRPO](https://huggingface.co/Joshua1702/fairrecovery-Qwen2.5-7B-GRPO)  
    *Best for complex reasoning and near-perfect fairness (Equity: 0.912).*
*   **Efficient Agent (Llama-1B)**: [Joshua1702/fairrecovery-Llama-3.2-1B](https://huggingface.co/Joshua1702/fairrecovery-Llama-3.2-1B)  
    *Best for low-latency edge deployment with strong equity performance (Equity: 0.840).*

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
| `R_fair` | −(avg_svc_normal − avg_svc_vuln) | 1.0 | Don't leave vulnerable zones behind |
| `R_safe` | −0.1 × violations | 0.5 | Respect constraints |
| `R_analysis` | Overlap(chosen, top-k) | 0.1 | Identify critical zones |
| **Terminal bonus**| 0.5×avg_svc + 0.5×(1+R_fair) | — | Long-horizon outcome |

*Fairness is normalized and bounded to prevent saturation and reward hacking.*

**Grader score: `0.6 × avg_service + 0.4 × (1 + fairness)` clamped to (0.01, 0.99)**

### Anti-Reward-Hacking Measures
- Stage ordering enforced (can't skip analyze → go straight to execute)
- Budget overflow: allocations rejected + penalty, state NOT mutated
- Persistent ignore penalty: if vulnerable zones receive 0 resources for 2+ consecutive days
- Early-submit blocked until MIN_STEPS reached
- Step cap: force-terminate at MAX_STEPS_SAFETY_CAP

**These safeguards ensure the agent cannot exploit reward signals without solving the actual task.**

---

## 📊 Visual Evidence Dashboard

In this section, we highlight the most critical performance metrics and fairness improvements achieved during training. These plots provide empirical proof of the agent's ability to navigate the fairness-utility trade-off.

### 📉 Training Evidence (Mandatory)

**All results are generated from real GRPO training runs (not synthetic or static evaluation).**

We trained using GRPO with Unsloth acceleration.

- **Models:** Llama-3.2-1B & Qwen-2.5-7B (4-bit)
- **Iteration Strategy:** We intentionally used a smaller model (Llama-1B) for rapid iteration and stable learning, then scaled to Qwen-7B for final performance.
- **Outcome:** Loss decreased steadily over time while reward converged.

📊 **Training Loss (Curriculum)**
![Training Loss](https://huggingface.co/spaces/Joshua1702/FairRecovery-PlusPlus/resolve/main/evidence/plots/training_loss.png)
*Figure: The agent steadily converges its policy over the curriculum.*

### 🏆 The Winning Metric: Dual-Model Comparison

![Model Comparison](https://huggingface.co/spaces/Joshua1702/FairRecovery-PlusPlus/resolve/main/evidence/plots/model_comparison.png)
*Figure 1: Comparison between Baseline, Llama-1B, and Qwen-7B. Our Premium Qwen-7B agent reaches a near-perfect **0.912 Equity Index**.*

### 📈 Training & Strategy Results

**Baseline vs Trained (Qwen)**
![Training Results](https://huggingface.co/spaces/Joshua1702/FairRecovery-PlusPlus/resolve/main/evidence/plots/training_results.png)
*Figure 2: 57% reward improvement.*

**Reward Heatmap**
![Score Heatmap](https://huggingface.co/spaces/Joshua1702/FairRecovery-PlusPlus/resolve/main/evidence/plots/score_heatmap.png)
*Figure 3: Consistency across episodes.*

**Fairness-Utility Frontier**
![Utility vs Fairness](https://huggingface.co/spaces/Joshua1702/FairRecovery-PlusPlus/resolve/main/evidence/plots/utility_vs_fairness.png)
**Extended Training Metrics**
![Extra Metrics](https://huggingface.co/spaces/Joshua1702/FairRecovery-PlusPlus/resolve/main/evidence/plots/training_metrics_extra.png)
*Figure 5: Detailed convergence and stability metrics.*

### ⏱️ Execution Dynamics (Step-by-Step)

![Fairness Improvement](https://huggingface.co/spaces/Joshua1702/FairRecovery-PlusPlus/resolve/main/evidence/plots/fairness_vs_episode.png)
*Figure 6: Global Fairness achievement trend over 32 training cycles.*

![Component Rewards](https://huggingface.co/spaces/Joshua1702/FairRecovery-PlusPlus/resolve/main/evidence/plots/component_rewards.png)
*Figure 7: Decomposed reward components showing the sacrifice of immediate utility for long-term equity.*

![Reward vs Steps](https://huggingface.co/spaces/Joshua1702/FairRecovery-PlusPlus/resolve/main/evidence/plots/reward_vs_steps.png)
*Figure 8: Performance stability during the 10-day recovery window.*

![Fairness vs Steps](https://huggingface.co/spaces/Joshua1702/FairRecovery-PlusPlus/resolve/main/evidence/plots/fairness_vs_steps.png)
*Figure 9: Cumulative equity growth per action.*

---


### Key Numbers

| Metric | Greedy Baseline | Llama-3.2-1B | Qwen-2.5-7B-GRPO |
|--------|----------------|--------------|------------------|
| Avg Episode Reward | 0.548 | 0.785 | **0.864** (+57%) |
| Avg Final Fairness | 0.732 | 0.840 | **0.912** (+24%) |
| Avg Final Utility  | 0.545 | 0.712 | **0.808** (+48%) |
| Strategy discovered | Always Zone 0 | Vulnerable-first | Strategic-equitable |

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

## 📁 Project Layout & Resources

### Repository Structure
```
fairrecovery_env/         # 🧠 Core OpenEnv Engine
├── constants.py          # Reward weights & thresholds
├── models.py             # Pydantic v2 Action/Obs/State
├── rewards.py            # 5-component RLVR reward engine
├── rubrics.py            # RFC 004 composable rubrics
└── ...                  # CityState, Tasks, Shield validator

server/                   # 🚀 Deployment & UI
├── app.py                # FastAPI + Gradio Web Dashboard
└── fairrecovery_env.py   # OpenEnv Environment Wrapper

asset_final/              # 📊 Final Evidence & Model Adapters
├── plots/                # Consolidated training visualizations
└── model/                # LoRA adapter weights (Qwen-7B)

docs/                     # 📝 Documentation & Reports
├── train_llama_final.ipynb # Llama-1B GRPO Notebook
└── train_qwen_final.ipynb  # Qwen-7B GRPO Notebook
```

### 🔗 Key Materials

| Resource | Link |
|----------|------|
| 🤗 **Live Space** | [Joshua1702/FairRecovery-PlusPlus](https://huggingface.co/spaces/Joshua1702/FairRecovery-PlusPlus) |
| 💻 **GitHub** | [joshua400/FairRecovery-PlusPlus](https://github.com/joshua400/FairRecovery-PlusPlus) |
| 📓 **Llama Training** | [Notebook](docs/train_llama_final.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshua400/FairRecovery-PlusPlus/blob/main/docs/train_llama_final.ipynb) |
| 📓 **Qwen Training** | [Notebook](docs/train_qwen_final.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshua400/FairRecovery-PlusPlus/blob/main/docs/train_qwen_final.ipynb) |
| 🧪 **Quick Testing (1.5B)** | [Kaggle Notebook](https://www.kaggle.com/code/joshuaragiland/fairevlo-lite) |
| 📝 **HF Blog Post** | [blog.md](blog.md) |

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
- ✅ GRPO training with TRL + Unsloth (see `docs/train_qwen_final.ipynb` and `docs/train_llama_final.ipynb`)
- ✅ Multi-Model evidence: Llama-1B & Qwen-7B plots in `evidence/plots/`
- ✅ Episode data in `episode_log.csv`
- ✅ Composable rubrics (OpenEnv RFC 004)
- ✅ Anti-reward-hacking: stage gates + persistent ignore penalty

---

## 🌍 Why This Matters

Disaster recovery systems often optimize speed, but ignore vulnerable populations.

FairRecovery++ shows that AI can:
- make fair decisions under constraints  
- balance efficiency and equity  
- improve real-world disaster planning  

👉 This opens the door to safer, fairer AI systems.

## 🎬 What the Agent Learned

Initially, the agent behaves like a human under pressure: **fix what is easiest first.**

After training, it learns something deeper: **true recovery is not about speed — it’s about fairness.**

This shift is what FairRecovery++ enables.

---

*Built for the Meta PyTorch OpenEnv Hackathon India 2026.*
