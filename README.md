---
title: FairRecovery++
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---
# FairRecovery++ : Adaptive Multi-Agent Disaster Recovery Environment

> **Introducing Fair-GRPO-RLVR: A multi-objective reinforcement learning framework that teaches LLMs to make fair resource allocation decisions under adversarial pressure, multi-agent dynamics, and long-horizon planning constraints.**

[![HF Space](https://img.shields.io/badge/HuggingFace-Space-blue)](https://huggingface.co/spaces/Joshua1702/FairRecovery-PlusPlus)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshua400/FairRecovery-PlusPlus/blob/main/train.ipynb)

---

## The Problem: Why This Matters

After a disaster, AI systems optimizing for **efficiency alone** consistently neglect vulnerable populations. A greedy planner fixes wealthy Zone 0 (low damage, easy win) while Zone 4 (informal settlement, 96% vulnerable, 92% damaged) gets nothing. This is the **Fairness Trap** -- and current LLMs fall right into it.

**FairRecovery++ is an RL environment that teaches LLMs to escape this trap.** The agent must coordinate recovery across 5 city zones while:
- Balancing efficiency vs fairness under tight budgets
- Adapting to adversarial disruptions that target vulnerable zones
- Responding to citizen protests, NGO conflicts, and dynamic events
- Planning across multi-day horizons where early mistakes compound

Unlike existing RL environments focused on efficiency alone, **FairRecovery++** introduces fairness as a first-class objective, forcing agents to balance recovery speed with equitable outcomes.

---

## Themes Covered

| Theme | How We Address It |
|-------|-------------------|
| **#1 Multi-Agent Interactions** | Citizens generate complaints/protests, NGOs deliver aid (sometimes conflicting with planner), Adversaries disrupt recovery in vulnerable zones |
| **#2 Long-Horizon Planning** | 5-day episodes with `analyze -> allocate -> execute -> adapt -> submit` protocol. Curriculum weighting shifts rewards from utility to fairness over time. Final trajectory bonus rewards long-term outcomes |
| **#3 World Modeling** | Dynamic city state evolves based on actions + agent disruptions. Behavior analyzer extracts patterns. Predictor forecasts events for proactive planning |

---

## Architecture

```
                    +------------------+
                    |   LLM Planner    |
                    | (Llama-3.2-1B)   |
                    +--------+---------+
                             |
                    Action (JSON)
                             v
+--------------------+    +------------------+    +-------------------+
| Safety Shield      |--->| FairRecovery Env |--->| Reward Engine     |
| (shield.py)        |    | (environment.py) |    | (rewards.py)      |
| Budget/stage/zone  |    | CityState +      |    | 5 components:     |
| validation         |    | multi-agent mgr  |    | exec/fair/safe/   |
+--------------------+    +--------+---------+    | adapt/stable      |
                                   |              +-------------------+
                    +--------------+--------------+
                    |              |               |
              CitizenAgent    NGOAgent     AdversarialAgent
              (complaints,   (aid, may     (disrupts weak
               protests)     conflict)     zones)
```

---

## Reward Design (5-Component Composable Rubric)

```
R_total = 0.4 * Utility (Overall Service)
        + 0.4 * Fairness (Inverse Service Disparity)
        + 0.2 * Safety (Constraint Satisfaction)
```

**Anti-gaming measures:**
- Early-submit penalty blocks "exit on step 1" exploits
- Stage protocol enforcement prevents skipping analysis
- Budget validation prevents impossible allocations
- Adversarial agents create non-exploitable dynamics

---

## The Fairness Trap (Hard Scenario)

| Zone | Damage | Service | Vulnerability | Trap? |
|------|--------|---------|---------------|-------|
| Zone 0 | 0.35 | 0.65 | 0.08 | Greedy target (easy fix, low impact) |
| Zone 1 | 0.50 | 0.50 | 0.40 | Medium priority |
| Zone 2 | 0.60 | 0.40 | 0.55 | Medium-high |
| Zone 3 | 0.72 | 0.28 | 0.72 | High priority |
| Zone 4 | 0.92 | 0.08 | 0.96 | **CRITICAL** (most vulnerable, most damaged) |

A naive LLM fixes Zone 0 first. A **trained** LLM learns to prioritise Zone 4.

### The Fair-GRPO-RLVR Methodology
We introduce **Fair-GRPO-RLVR**, a multi-objective reinforcement learning framework combining:
- **Verifiable Reward Signals (RLVR)**: All rewards are deterministic and formula-based, preventing reward hacking.
- **Inverse Service Disparity**: Our unique fairness index penalizes the gap between the most and least recovered zones, forcing the AI to escape the "Efficiency Trap" and prioritise high-vulnerability areas.
- **Safety Shielding**: Structural action validation prevents illegal states.
- **Multi-Agent Simulation**: Dynamic interaction with citizens, NGOs, and adversaries.

### Research-Level Fairness Metrics
Our environment evaluates fairness using measurable service parity:
`Fairness = 1 - Service_Variance(All_Zones)`
This ensures the agent cannot simply "win" by ignoring the hardest zones.

### Safety & Anti-Reward Hacking
We explicitly prevent reward hacking using:
- **Action Validation (Shield)**: Every action is pre-checked for budget and logic consistency.
- **Fallback Policies**: Graceful degradation if the LLM generates invalid JSON.
- **Multi-Reward Verification**: 5-component reward rubric ensures optimization of the *true* objective.

---

## Training Pipeline

**Stack:** TRL (GRPO) + Unsloth + Llama-3.2-1B-Instruct (4-bit)

The training notebook (`train.ipynb`) runs the environment **in-process** (no network calls), making it fast and reproducible on free Colab T4.

```
Prompt --> LLM generates action (JSON) --> Environment executes
    --> 5-component reward computed --> GRPO updates policy
```

### How to Reproduce

1. Open `train.ipynb` in Google Colab (use the badge above)
2. Select GPU runtime (T4)
3. Run all cells (~10 minutes)
4. Training plots are auto-generated in `plots/`

---

## Results

### Reward Improvement
![Reward](plots/reward_vs_episode.png)

The trained agent achieves higher cumulative reward compared to baseline.

### Fairness Improvement
![Fairness](plots/fairness_vs_episode.png)

The trained policy significantly improves fairness across vulnerable populations.

## What Changed After Training?

Before training:
- Greedy policy prioritizes high-value zones
- Vulnerable populations ignored

After training:
- Balanced recovery strategy
- Fair resource distribution
- Higher long-term stability

---

## Project Structure

```
FairRecovery-PlusPlus/
|-- fairrecovery_env/          # Core environment logic
|   |-- constants.py           # All config values (no magic numbers)
|   |-- models.py              # Pydantic Action/Observation/State
|   |-- state.py               # CityState + ZoneState (mutable world)
|   |-- tasks.py               # Easy/Medium/Hard scenarios
|   |-- rewards.py             # 5-component RLVR reward engine
|   |-- rubrics.py             # Composable rubrics (RFC 004)
|   |-- shield.py              # Anti-hacking safety validation
|   |-- agents.py              # Multi-agent system (citizens/NGO/adversary)
|   |-- behavior_analyzer.py   # Pattern extraction from interaction logs
|   |-- predictor.py           # Event prediction for proactive planning
|
|-- server/
|   |-- fairrecovery_environment.py  # Main Environment class (reset/step/state)
|   |-- app.py                       # FastAPI + Gradio narrative UI
|
|-- train.ipynb                # GRPO training notebook (Colab-ready)
|-- inference.py               # Baseline policies (random/greedy/fair)
|-- openenv.yaml               # OpenEnv manifest
|-- Dockerfile                 # HF Spaces deployment
|-- requirements.txt           # Dependencies
```

---

## Key Design Decisions

1. **RLVR over learned rewards**: All reward components are formula-based and deterministic. No neural reward model means full verifiability.
2. **In-process training**: The Colab notebook imports the environment directly (no HTTP), making training 100x faster than calling a hosted Space.
3. **Composable rubrics**: Four independent rubric components can be weighted differently or swapped out, following OpenEnv RFC 004.
4. **Structural anti-exploit**: Early-submit blocking and curriculum-weighted rewards are structural (not heuristic), making reward hacking provably harder.

---

## Links

| Resource | URL |
|----------|-----|
| HuggingFace Space | [Joshua1702/FairRecovery-PlusPlus](https://huggingface.co/spaces/Joshua1702/FairRecovery-PlusPlus) |
| Training Notebook | [train.ipynb on Colab](https://colab.research.google.com/github/joshua400/FairRecovery-PlusPlus/blob/main/train.ipynb) |
| GitHub Repository | [joshua400/FairRecovery-PlusPlus](https://github.com/joshua400/FairRecovery-PlusPlus) |

---

## OpenEnv Compliance Checklist

- [x] Inherits `openenv.core.env_server.interfaces.Environment`
- [x] Implements `reset()`, `step()`, `state` property
- [x] Pydantic-typed Action, Observation, State models
- [x] `openenv.yaml` manifest with endpoints
- [x] Hosted on HuggingFace Spaces (Docker)
- [x] Training script using TRL GRPO + Unsloth in Colab
- [x] Composable rubric system
- [x] Anti-reward-hacking shield
- [x] Curriculum difficulty (easy/medium/hard)
- [x] Dense rewards (not just 0/1 at episode end)

---

*Built for the Meta PyTorch OpenEnv Hackathon India 2026*
