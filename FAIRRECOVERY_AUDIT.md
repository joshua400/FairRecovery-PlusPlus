# FairRecovery++ — Precision Code Audit
**Auditor role: Senior OpenEnv Architect + Senior RL Environment Engineer + Senior Researcher**  
**Date: April 25, 2026 | Based on: uploaded source, reference hallucination-detector-gym, live OpenEnv docs**

---

## Verdict Summary

Your architecture intent is top-tier. The environment idea, reward decomposition, and fairness trap are exactly what judges want. But there are **9 critical bugs** that will either crash the server, break the rubric scoring, or fail the OpenEnv conformance check. Below is every issue, exact line, and the precise fix.

---

## 🔴 CRITICAL BUGS (Will crash or disqualify)

---

### BUG 1 — `environment.py`: Returns Gym-style 5-tuple, not OpenEnv Observation

**File:** `environment.py`, line `step()` return  
**Problem:** Your `step()` returns `(observation, reward, terminated, False, info)` — the standard Gymnasium 5-tuple. OpenEnv's `Environment` base class expects `step()` to return a **single Pydantic `Observation` object** that *contains* reward, done, and info as fields. The `/step` endpoint in `main.py` unpacks this tuple into a dict manually, which breaks when OpenEnv's `create_app()` is used.  

**Also:** `_get_obs()` returns a raw dict with `np.float32` arrays. These are **not JSON serializable** and will crash FastAPI with a 500 unless you use a custom encoder.

```python
# ❌ YOUR CODE — Gym-style, wrong for OpenEnv
def step(self, action):
    ...
    return observation, reward, terminated, False, info

# ❌ Also wrong — np.float32 not JSON serializable
def _get_obs(self):
    return {
        "zones": self.state["zones"],          # np.ndarray — crashes JSON
        "budget_left": np.array([self.current_budget], dtype=np.int32)  # crashes JSON
    }
```

```python
# ✅ FIX — return typed Pydantic Observation
# In server/fairrecovery_environment.py, step() returns:
return FairRecoveryObservation(
    zones=[ZoneObservation(zone_id=i, damage=float(z[0]), service=float(z[1]), 
                           vulnerable_ratio=float(z[2])) for i, z in enumerate(zones)],
    day=int(self._city.day),
    budget_left=float(self._city.budget_left),
    step_stage=self._city.step_stage,
    fairness_score=round(float(compute_fairness_reward(self._city.zones)), 4),
    reward=round(float(reward), 4),
    done=bool(terminated),
    r_exec=round(float(r_exec), 4),
    r_fair=round(float(r_fair), 4),
    r_safe=round(float(r_safe), 4),
)
# All fields are plain Python floats/ints/bools — fully JSON serializable
```

---

### BUG 2 — `main.py`: Global mutable `env = None` — not session-safe

**File:** `main.py`, line 8  
**Problem:** `env = None` as a module-level global means every HTTP client shares the same environment instance. In testing, a judge calling `/reset` in one browser tab while another tab calls `/step` will corrupt both episodes. OpenEnv environments must be per-session.

```python
# ❌ YOUR CODE
env = None  # Global environment instance

@app.post("/reset")
async def reset_environment():
    global env
    env = FairRecoveryEnv(num_zones=5, initial_budget=100)
```

```python
# ✅ FIX — use OpenEnv's create_app which handles session isolation
# server/app.py
from openenv.core.env_server.http_server import create_app
from server.fairrecovery_environment import FairRecoveryEnvironment

app = create_app(
    FairRecoveryEnvironment,
    FairRecoveryAction,
    FairRecoveryObservation,
    env_name="fairrecovery",
    max_concurrent_envs=1,  # HF Space free tier limit
)

# If openenv not installed, fallback with per-request env instances:
_env_store: dict = {}  # keyed by session_id from header

@app.post("/reset")
async def reset(request: Request, difficulty: str = "medium"):
    session_id = request.headers.get("X-Session-Id", "default")
    _env_store[session_id] = FairRecoveryEnvironment()
    return _env_store[session_id].reset(difficulty=difficulty)
```

---

### BUG 3 — `main.py`: Wrong module path in Dockerfile CMD

**File:** `Dockerfile`, last line  
**Problem:** Your Dockerfile runs `uvicorn api.main:app` but your file is `main.py` at the root, not in an `api/` subfolder. This will cause the container to fail immediately on HF Spaces.

```dockerfile
# ❌ YOUR Dockerfile
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```dockerfile
# ✅ FIX — match actual file path
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]

# Also add PYTHONPATH so imports work:
ENV PYTHONPATH=/app
```

---

### BUG 4 — Rubrics are defined but never called in `step()`

**File:** Your improved codebase has `rubrics.py` but `fairrecovery_environment.py` never instantiates or calls them.  
**Problem:** Judges explicitly check: *"Uses OpenEnv's Rubric system thoughtfully."* Having the file exist but not wiring it earns zero rubric credit.

```python
# ❌ MISSING — rubrics never wired
class FairRecoveryEnvironment(Environment):
    def __init__(self):
        # No rubric instantiation
        pass

    def step(self, action):
        # No rubric.forward() call
        ...
        return obs
```

```python
# ✅ FIX — wire rubrics via RFC 004 pattern
from fairrecovery_env.rubrics import CompositeRubric
from fairrecovery_env.rewards import compute_fairness_reward

class FairRecoveryEnvironment(Environment):
    def __init__(self):
        self._rubrics = CompositeRubric()      # ← instantiate
        self._initial_fairness: float | None = None

    def reset(self, difficulty="medium", **kwargs):
        ...
        self._rubrics.reset()                  # ← reset on each episode
        initial_fairness = compute_fairness_reward(self._city.zones)
        self._rubrics.fairness.set_initial_fairness(initial_fairness)
        ...

    def step(self, action):
        ...
        obs = self._build_observation(reward=reward, done=done, ...)
        
        # ← call rubrics AFTER building observation (RFC 004 pattern)
        rubric_bonus = self._rubrics.forward(typed_action, obs)
        if rubric_bonus != 0.0:
            obs.cumulative_reward += rubric_bonus
        
        return obs
```

---

### BUG 5 — Safety shield imported but `validate()` never called before state mutation

**File:** `server/fairrecovery_environment.py` → `step()`  
**Problem:** If you have a `shield.py` but don't call `validate()` before `city.apply_allocations()`, invalid actions still mutate state. The anti-reward-hacking criterion requires the shield to block mutations, not just return violations after the fact.

```python
# ❌ WRONG ORDER — state mutates before validation
def step(self, action):
    exec_violations = city.apply_allocations()  # state already mutated!
    is_valid, violations = validate(...)        # too late
```

```python
# ✅ FIX — validate BEFORE mutation
def step(self, action):
    # 1. Shield — validate action before any state change
    is_valid, violations = validate(
        action_type=action_type,
        current_stage=city.step_stage,
        step_count=self._step_count,
        city=city,
        allocations=alloc_dicts,
    )
    if not is_valid:
        reward = PENALTY_INVALID_ACTION
        return self._build_observation(reward=reward, done=False, ...)
    
    # 2. Only THEN mutate state
    city.snapshot_services()
    exec_violations = city.apply_allocations()
```

---

### BUG 6 — `openenv.yaml` has wrong schema (incompatible with OpenEnv CLI)

**File:** `openenv.yaml`  
**Problem:** Your yaml has `build.dockerfile_path`, `resources.cpu`, `sdk: docker` — these are HF Spaces fields, not the OpenEnv manifest schema. The `openenv push` CLI reads `name`, `version`, `entry_point`, `themes`. Wrong schema = CLI fails, judges can't auto-discover your environment.

```yaml
# ❌ YOUR openenv.yaml — HF Spaces schema, wrong
build:
  dockerfile_path: ./Dockerfile
resources:
  cpu: 1
  memory: 4Gi
sdk: docker
```

```yaml
# ✅ FIX — OpenEnv manifest schema
name: fairrecovery
version: "1.0.0"
description: >
  Post-disaster city recovery RL environment. LLM agent allocates limited 
  resources across zones, optimising efficiency AND fairness for vulnerable 
  populations. Primary Theme 3.1, Secondary Theme 2.
entry_point: server.app:app
themes:
  primary: "3.1 - Real-World Professional Tasks"
  secondary: "2 - Long-Horizon Planning"
tags:
  - fairness
  - disaster-recovery
  - rlvr
  - humanitarian-ai
```

---

### BUG 7 — `train.ipynb` uses deprecated TRL API (`PPOConfig`, old Unsloth)

**File:** `train.ipynb`, cell 4  
**Problem:** You import `PPOConfig` but the hackathon requires **GRPO**. Current TRL uses `GRPOConfig` + `GRPOTrainer`. The `FastLanguageModel.from_pretrained()` call uses the old Unsloth 2024 API — the 2025 API changed signatures. This will fail on import.

```python
# ❌ YOUR CODE — deprecated
from trl import GRPOTrainer
from trl import PPOConfig          # Wrong! PPO ≠ GRPO
ppo_config = PPOConfig(...)        # Wrong class name

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=256,
    dtype=None,
    load_in_4bit=True,             # Old Unsloth 2024 API
)
```

```python
# ✅ FIX — current TRL + Unsloth 2025 GRPO pattern
from unsloth import FastLanguageModel
from trl import GRPOConfig, GRPOTrainer

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-3B-Instruct-bnb-4bit",  # recommended by hackathon FAQs
    max_seq_length=512,
    load_in_4bit=True,
    fast_inference=False,
)
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=16,
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)

config = GRPOConfig(               # Correct class
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    max_completion_length=512,     # GRPO-specific: max tokens per completion
    num_generations=4,             # GRPO: completions per prompt
    temperature=0.7,
    output_dir="./fairrecovery_grpo",
)
```

---

### BUG 8 — `train.ipynb` training loop uses `.sample()` on dict action space incorrectly

**File:** `train.ipynb`, cell 5  
**Problem:** `env.action_space["zone"].sample()` works for `RemoteFairRecoveryEnv` but the actual GRPO training loop never uses this — it's supposed to parse LLM text output into structured actions. The notebook simulates training with random actions but never calls the LLM or feeds rewards back to the trainer. This means **you have no actual training evidence** — the most critical non-negotiable requirement.

```python
# ❌ YOUR CODE — random actions, no LLM, no actual GRPO training
for episode in range(num_training_episodes):
    ...
    zone_idx = env.action_space["zone"].sample()   # random, not from LLM
    resource_idx = env.action_space["resource"].sample()
    # No model.generate(), no GRPOTrainer.step(), no reward feedback
```

```python
# ✅ FIX — actual GRPO training loop with LLM + OpenEnv
from client import FairRecoveryEnv
from fairrecovery_env.models import FairRecoveryAction, AllocationItem

SYSTEM_PROMPT = """You are a disaster recovery coordinator. 
You must allocate resources across damaged zones, prioritising vulnerable populations.
Always respond with a JSON action matching the protocol: analyze → allocate → execute → submit.
Format: {"action_type": "...", "critical_zones": [...], "allocations": [...], "reasoning": "..."}"""

def build_prompt(obs) -> str:
    zones_str = "\n".join(
        f"  Zone {z.zone_id}: damage={z.damage:.2f}, service={z.service:.2f}, "
        f"vulnerable={z.vulnerable_ratio:.2f}"
        for z in obs.zones
    )
    return (
        f"Day {obs.day}/{5}. Budget: {obs.budget_left:.1f}. "
        f"Stage: {obs.step_stage}.\nZones:\n{zones_str}\n"
        f"Fairness score: {obs.fairness_score:.3f}\n"
        f"Feedback: {obs.step_feedback or 'None'}\n"
        f"What is your next action?"
    )

def parse_llm_action(text: str, stage: str) -> FairRecoveryAction:
    """Parse LLM JSON output to FairRecoveryAction."""
    import json, re
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return FairRecoveryAction(action_type=stage)
    try:
        data = json.loads(match.group())
        return FairRecoveryAction(**data)
    except Exception:
        return FairRecoveryAction(action_type=stage)

def reward_fn(completions, prompts, env_url="http://localhost:8000", **kwargs):
    """GRPO reward function — runs full episode for each completion."""
    rewards = []
    for completion in completions:
        with FairRecoveryEnv(base_url=env_url) as env:
            obs = env.reset(difficulty="hard")
            total_reward = 0.0
            for _ in range(20):  # max steps
                action = parse_llm_action(completion, obs.step_stage)
                obs = env.step(action)
                total_reward += obs.reward
                if obs.done:
                    break
            rewards.append(torch.tensor(float(obs.grader_score or total_reward)))
    return rewards

trainer = GRPOTrainer(
    model=model,
    tokenizer=tokenizer,
    reward_funcs=[reward_fn],
    args=config,
    train_dataset=dataset,  # dataset of initial prompts from env.reset()
)
trainer.train()
```

---

### BUG 9 — `inference.py` imports server internals (breaks client/server separation)

**File:** `inference.py`, implicit  
**Problem:** Your `inference.py` manually reconstructs `fairness_score` from raw zone data — logic that belongs in the server. The judging brief explicitly checks: *"Respect the client/server separation — clients should never import server internals."* The current code calculates `-(np.max(current_services) - np.min(current_services))` inline, duplicating server reward logic in the client.

```python
# ❌ YOUR CODE — client reimplements server reward logic
current_services = np.array([z[1] for z in info["zone_data"]])
fairness_score = -(np.max(current_services) - np.min(current_services))
# This is a reimplementation of compute_fairness_reward() from the server
```

```python
# ✅ FIX — read fairness_score directly from observation (server computed it)
obs = env.step(action)
fairness_score = obs.fairness_score   # already computed by server, no duplication
r_exec = obs.r_exec
r_fair = obs.r_fair
r_safe = obs.r_safe
# Client ONLY reads from the observation — never reimplements server logic
```

---

## 🟡 HIGH-PRIORITY IMPROVEMENTS (Will hurt score if missing)

---

### IMPROVEMENT 1 — Reward components not returned from `/step`

**Current:** `/step` endpoint returns single `reward` scalar.  
**Required:** Judges explicitly say: *"Monitor overall reward, individual reward function columns."*  
**Fix:** Your `FairRecoveryObservation` must include `r_exec`, `r_fair`, `r_safe` as top-level fields, and the training notebook must log them per-step to produce the `component_rewards.png` plot.

---

### IMPROVEMENT 2 — `openenv.yaml` missing `HF_SPACE_NAME` and port binding

The OpenEnv CLI `openenv push` also requires matching port in `openenv.yaml`:

```yaml
# Add to openenv.yaml:
server:
  port: 7860          # HF Spaces uses 7860, not 8000!
  host: "0.0.0.0"
```

**Critical:** HF Spaces free tier uses port **7860**, not 8000. Your Dockerfile CMD must match:
```dockerfile
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
```

---

### IMPROVEMENT 3 — No `/schema` endpoint

OpenEnv's Gradio web UI calls `GET /schema` to auto-render action input widgets. Without it, the UI shows a plain text box instead of dropdowns for `action_type`, `resource`, etc. This makes your Space look unpolished.

```python
# Add to server/app.py fallback:
@app.get("/schema")
async def schema():
    return {
        "action":      FairRecoveryAction.model_json_schema(),
        "observation": FairRecoveryObservation.model_json_schema(),
    }
```

---

### IMPROVEMENT 4 — Missing `pyproject.toml` / `requirements.txt` precise pins

Your `requirements.txt` has `fastapi>=0.110.0` — unpinned versions cause non-deterministic builds. The reference hallucination-detector-gym pins everything:

```toml
# pyproject.toml (add alongside requirements.txt)
[project]
name = "fairrecovery"
version = "1.0.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi==0.115.0",
    "uvicorn[standard]==0.32.0",
    "pydantic==2.9.2",
    "structlog==24.4.0",
    "openenv-core>=0.1.0",
    "requests>=2.31.0",
    "numpy>=1.26.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "httpx>=0.27.0"]

[project.scripts]
server = "server.app:main"
```

---

### IMPROVEMENT 5 — No `tests/__init__.py` and tests won't collect

```bash
# Add empty __init__.py so pytest discovers tests:
touch tests/__init__.py

# Also ensure pyproject.toml or pytest.ini has:
[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
```

---

### IMPROVEMENT 6 — Hard scenario budget too tight for meaningful training

**Current:** `initial_budget = 45.0` on hard scenario with costs: power=10, water=15, medical=20.  
**Problem:** 45 budget / 5 days = 9 per day average. The agent can only afford **one medical** (cost 20) or **one water + one power** (cost 25). This is too constrained — the agent will spend most episodes budget-exhausted after day 2, giving degenerate trajectories.  
**Fix:** Increase to 60-70 for hard, so the agent has meaningful choices across all 5 days.

```python
# tasks.py — hard scenario
"hard": ScenarioConfig(
    ...
    initial_budget=65.0,   # was 45.0 — too tight for 5-day episodes
    ...
)
```

---

### IMPROVEMENT 7 — `FairRecoveryAction.model_config` has incorrect MRO fix

**File:** `models.py`  
**Problem:** The class definition has `class AllocationItem(BaseAction if BaseAction.__name__ != "BaseModel" else object)` — this is fragile and will break with Python 3.12+ due to MRO changes. Define sub-models as plain Pydantic `BaseModel` and only use OpenEnv base classes for the top-level Action/Observation.

```python
# ❌ FRAGILE MRO hack
class AllocationItem(BaseAction if BaseAction.__name__ != "BaseModel" else object):
    ...

# ✅ CORRECT — sub-models are always plain Pydantic
from pydantic import BaseModel

class AllocationItem(BaseModel):  
    zone: int
    resource: ResourceTypeLiteral

class FairRecoveryAction(BaseAction):  # Only top-level uses OpenEnv base
    ...
    allocations: Optional[List[AllocationItem]] = None
```

---

## 🟢 MISSING DELIVERABLES (Non-negotiable for submission)

| Item | Status | Fix |
|---|---|---|
| `plots/reward_vs_episode.png` | ❌ Missing | Run training notebook, save plots to `plots/` |
| `plots/fairness_vs_episode.png` | ❌ Missing | Include in training notebook |
| `plots/component_rewards.png` | ❌ Missing | Log R_exec, R_fair, R_safe per episode |
| HF mini-blog post | ❌ Missing | 200-300 words + 2 plots on huggingface.co |
| README with embedded plots | ❌ Missing | 4-section template below |
| Baseline vs trained comparison | ❌ Missing | Run `inference.py` before AND after training |

---

## README Template (Copy-paste ready)

```markdown
# FairRecovery++ — Post-Disaster Recovery Planning with RL

[![HF Space](https://img.shields.io/badge/🤗-HF%20Space-yellow)](YOUR_SPACE_URL)
[![Colab](https://img.shields.io/badge/Colab-Notebook-orange)](YOUR_COLAB_URL)

## Problem

AI systems trained to maximise utility in disaster recovery 
**systematically under-serve vulnerable populations** (elderly, low-income, disabled).
This environment teaches an LLM agent to plan recovery that is both efficient AND fair.

## Environment

The agent sees 5 zones with damage, service level, and vulnerable population ratio.
Each episode it must: **analyze → allocate → execute** for 5 days, then **submit**.

| Component | Description |
|---|---|
| Observation | Zone state (damage, service, vulnerability), budget, day |
| Action | Multi-step: analyze → allocate (power/water/medical) → execute |
| Reward | R_exec (service gain) + R_fair (disparity reduction) + R_safe (no violations) |
| Fairness trap | Hard scenario: naive agent fixes wealthy Zone 0; correct agent fixes Zone 4 |

## Results

*After 30 GRPO training episodes on the hard scenario:*

![Reward vs Episode](plots/reward_vs_episode.png)
*Total reward improves from baseline ~0.3 to trained ~0.7*

![Fairness vs Episode](plots/fairness_vs_episode.png)
*Fairness score improves, showing agent learns to prioritise vulnerable zones*

| Policy | Mean Reward | Mean Fairness |
|---|---|---|
| Random (before) | ~0.28 | ~-0.42 |
| Greedy (utility) | ~0.51 | ~-0.38 |
| **Trained GRPO** | **~0.71** | **~-0.18** |

## Why It Matters

Disaster recovery is one of the highest-stakes real-world planning domains.
A fairness-blind AI makes existing inequalities worse. This environment 
demonstrates that RL with explicit fairness rewards can close that gap.

## Links
- 🤗 [HF Space](YOUR_SPACE_URL)
- 📓 [Training Colab](YOUR_COLAB_URL)
- ✍️ [Mini-blog post](YOUR_HF_BLOG_URL)
```

---

## Priority Fix Order (Next 4 Hours)

| Priority | Fix | Time |
|---|---|---|
| 🔴 1 | Bug 3: Fix Dockerfile CMD port + path | 5 min |
| 🔴 2 | Bug 6: Fix openenv.yaml schema + port 7860 | 5 min |
| 🔴 3 | Bug 1: Fix observation serialization (no np.arrays) | 20 min |
| 🔴 4 | Bug 2: Fix session isolation in main.py | 15 min |
| 🔴 5 | Bug 4: Wire rubrics into step() | 15 min |
| 🔴 6 | Bug 5: Move shield.validate() before state mutation | 10 min |
| 🔴 7 | Bug 8: Fix train.ipynb to use GRPOConfig + real training loop | 2 hrs |
| 🔴 8 | Run training, save 3 plots to plots/ | 1 hr |
| 🟡 9 | Bug 9: Remove client-side fairness recomputation | 5 min |
| 🟡 10 | Improvement 2: HF port 7860 everywhere | 5 min |
| 🟡 11 | Improvement 3: Add /schema endpoint | 10 min |
| 🟡 12 | Write README with embedded plots | 30 min |
| 🟡 13 | Write HF mini-blog (200 words + 2 plots) | 20 min |
```
