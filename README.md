# FairRecovery++ — Adaptive Multi-Agent Post-Disaster Recovery Environment

<p align="center">
  <strong>An OpenEnv RL environment for training AI agents to coordinate equitable disaster recovery with dynamic multi-agent interactions</strong>
</p>

---

## Overview

**FairRecovery++** is a production-grade reinforcement learning environment designed for the [OpenEnv](https://openenv.ai) ecosystem. It simulates post-disaster city recovery where an AI planner must allocate limited resources across damaged zones while interacting with dynamic agents (citizens, NGOs, adversaries) and adapting to evolving behavioral patterns.

### Key Innovation: Multi-Agent Adaptive System

Unlike static RL environments, FairRecovery++ features:

| Component | Description |
|-----------|-------------|
| **Citizen Agents** | React to recovery progress with satisfaction decay, complaints, and protests |
| **NGO Agents** | Provide supplementary resources, may conflict with planner priorities |
| **Adversarial Agents** | Exploit weaknesses, disrupt recovery, exacerbate inequalities |
| **Behavior Analyzer** | Extracts patterns from interaction logs in real-time |
| **Predictive Engine** | Forecasts next events to enable proactive planning |

### Hackathon Theme Coverage

| Theme | Coverage | Implementation |
|-------|----------|----------------|
| **Theme 3.1** — Real-World Professional Tasks | ⭐ Core | Disaster recovery coordination with equity constraints |
| **Theme 2** — Long-Horizon Planning | ⭐ Strong | Multi-day episodes with cascading consequences |
| **Theme 1** — Multi-Agent Interaction | ⭐ Strong | Citizens, NGOs, adversaries with behavioral dynamics |

---

## Architecture

```
fairrecovery_env/            # Core environment package
├── constants.py             # All configuration — no magic numbers
├── models.py                # Pydantic v2 Action/Observation/State
├── state.py                 # Mutable world model (CityState/ZoneState)
├── tasks.py                 # 3 scenarios (easy/medium/hard + fairness trap)
├── rewards.py               # 5-component RLVR reward engine
├── rubrics.py               # RFC 004 composable rubrics
├── shield.py                # Safety validation before state mutation
├── agents.py                # Multi-agent system (Citizens, NGOs, Adversaries)
├── behavior_analyzer.py     # Behavioral pattern extraction
├── predictor.py             # Predictive response engine
└── logging_config.py        # Structured JSON logging

server/                      # HTTP API layer
├── fairrecovery_environment.py  # Core Environment class
└── app.py                   # FastAPI application

tests/                       # Comprehensive test suite
└── test_environment.py      # 30+ tests covering all components

client.py                    # Typed Python client
inference.py                 # Baseline policy evaluation
openenv.yaml                 # OpenEnv configuration
pyproject.toml               # Python project config
Dockerfile                   # Production container
```

---

## Reward System

Dense, verifiable, formula-based rewards (no learned reward model):

```
R_total = 0.30 × R_exec   (service improvement)
        + 0.25 × R_fair   (disparity reduction)
        + 0.20 × R_adapt  (adaptation to predicted events)
        + 0.15 × R_stable (citizen satisfaction balance)
        + 0.10 × R_safe   (constraint satisfaction)
```

### Anti-Exploitation Measures
- **Fairness trap** in hard scenario (Zone 0 is easy but low-priority)
- **Persistent ignore penalties** for neglecting vulnerable zones
- **Adversarial disruptions** that punish static strategies
- **Grader scores strictly in (0.01, 0.99)** — never exactly 0 or 1

---

## Quick Start

### Install & Run Server
```bash
pip install -e ".[dev]"
python -m server.app
```

### Run Baseline Inference
```bash
python inference.py --difficulty hard --episodes 5 --policy all
```

### Run Tests
```bash
pytest tests/ -v --tb=short
```

### Docker
```bash
docker build -t fairrecovery .
docker run -p 8000:8000 fairrecovery
```

---

## Training Integration

Compatible with TRL/GRPO/Unsloth training pipelines:

```python
from client import FairRecoveryEnv
from fairrecovery_env.models import FairRecoveryAction

env = FairRecoveryEnv(base_url="http://localhost:8000")
obs = env.reset(difficulty="hard")

# Your LLM policy generates actions based on observations
action = your_policy(obs)
obs = env.step(action)

# Dense reward at every step — 5 components for analysis
print(f"R_exec={obs.r_exec}, R_fair={obs.r_fair}, R_adapt={obs.r_adapt}")
```

---

## License

BSD-3-Clause
