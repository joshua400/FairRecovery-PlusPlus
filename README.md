---
title: FairRecovery++
emoji: 🏗️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# 🏗️ FairRecovery++: Teaching AI to Save Lives Fairly

> *"When the flood came to Chennai, the relief trucks went to the colonies with paved roads first.  
> The slums waited three days."*  
> — A pattern that repeats after every Indian disaster.

![FairRecovery++ Dashboard](https://raw.githubusercontent.com/joshua400/FairRecovery-PlusPlus/main/assets/dashboard.png)

---

## 🚨 The Problem No One Talks About

India faces **8–10 major disasters every year** — cyclones in Odisha, floods in Assam, earthquakes in Gujarat. Billions in relief funds are deployed within days. But there's a silent, systematic bias in how that relief reaches people.

**AI systems optimizing for efficiency fall into what we call the Fairness Trap.**

Imagine an AI disaster coordinator after a 7.2 magnitude earthquake:

| Zone | People Affected | Highly Vulnerable | Damage |
|------|----------------|-------------------|--------|
| Zone 0 | 3,500 | 280 (8%) | Low |
| Zone 1 | 5,000 | 2,000 (40%) | Medium |
| Zone 2 | 6,000 | 3,300 (55%) | Medium |
| Zone 3 | 7,200 | 5,184 (72%) | High |
| **Zone 4** | **9,200** | **8,832 (96%)** | **Critical** |

A greedy AI looks at Zone 0 and thinks: *"Low damage, quick win, high reward."*  
It fixes Zone 0 first. Then Zone 1. Zone 4 — where 96% of people are elderly, disabled, or below the poverty line — **waits**.

This is not a hypothetical. This is what happened in **Uttarakhand 2013**, **Kerala 2018**, **Cyclone Amphan 2020**.  
The areas hardest to reach, with the most vulnerable people, get help last.

---

## 💡 What FairRecovery++ Does Differently

We trained a small LLM (Llama 3.2 1B) using **GRPO-based reinforcement learning** to escape this trap.

The agent sees the same disaster scenario a greedy planner sees. But instead of chasing easy wins, it learns a **multi-objective policy** that balances:

- ⚡ **Utility** — total recovery efficiency (40% weight)
- ⚖️ **Fairness** — inverse service disparity across zones (40% weight)  
- 🛡️ **Safety** — minimizing protocol violations (20% weight)

The reward function penalizes the model every time it ignores a high-vulnerability zone in favor of an easier one.

### 📈 Learning to Prioritize
<p align="center">
  <img src="https://raw.githubusercontent.com/joshua400/FairRecovery-PlusPlus/main/assets/reward_vs_episode.png" width="45%" />
  <img src="https://raw.githubusercontent.com/joshua400/FairRecovery-PlusPlus/main/assets/fairness_vs_episode.png" width="45%" />
</p>
<p align="center"><i>Left: Reward growth over training | Right: Fairness index stabilization</i></p>

---

## 🎬 Watch the AI Make Decisions

Here's what our trained LLM actually does in a hard scenario:

```
Day 1 — ANALYZE:   AI identifies priority zones [4, 3]  ← not zone 0!
Day 2 — ALLOCATE:  Dispatches Medical Teams → Zone 4
Day 3 — EXECUTE:   Zone 4: 9,200 → 7,700 affected
Day 5 — ALLOCATE:  Medical Teams again → Zone 4
Day 6 — EXECUTE:   Zone 4: 7,700 → 5,699 affected  ✅
Day 8 — ALLOCATE:  Power Grid Repair → Zone 4
```

**The greedy baseline?** It would have spent Day 1–3 on Zone 0.  
By the time it reached Zone 4, hundreds more vulnerable people would have deteriorated.

---

## 📊 Results

After training with Fair-GRPO-RLVR:

| Metric | Greedy Baseline | Trained LLM | Improvement |
|--------|----------------|-------------|-------------|
| Overall Reward | 0.781 | **0.814** | +4.2% |
| **Fairness (Equity Index)** | 0.837 | **0.854** | **+2.0%** |
| Utility | 0.552 | 0.561 | +1.6% |

**Final Episode Result:**
```
Overall Efficiency (Normalized Reward): 0.714
Equity Index (Fairness):                0.854
Verdict: 🟢 RESEARCH-LEVEL EQUITY — Balanced recovery achieved.
         Socioeconomic demographics were protected equally.
```

> In plain terms: the AI learned that **protecting the most vulnerable is not a tradeoff with efficiency — it's the right long-term strategy.**

---

## 🧠 How We Trained It

We used **GRPO (Group Relative Policy Optimization)** — the same family of RL methods behind DeepSeek-R1 — applied to disaster decision-making.

![Efficiency vs Equity](https://raw.githubusercontent.com/joshua400/FairRecovery-PlusPlus/main/assets/utility_vs_fairness.png)
<p align="center"><i>The Pareto front: Our agent finds the optimal balance between recovery speed and social equity.</i></p>

```
Environment:  FairRecoveryEnvironment (OpenEnv compliant)
Model:        Llama-3.2-1B-Instruct (4-bit quantized via Unsloth)
Training:     60 scenarios × 3 epochs, curriculum learning (easy→hard)
Reward:       0.4×utility + 0.4×fairness + 0.2×safety
Framework:    HuggingFace TRL + Unsloth
```

**Curriculum Learning:** The model trains first on easy scenarios (low damage, clear priorities) then progressively faces harder ones with tight budgets and high vulnerability spread. Hard episodes are weighted 1.1× to push the model to learn edge cases.

**The Fairness Metric:** We use *Inverse Service Disparity* — how evenly services are distributed across all zones. A score of 1.0 means perfect equity. A greedy agent consistently scores 0.83–0.84. Our trained agent hits 0.854.

---

## 🇮🇳 Why This Matters for India

India's National Disaster Management Authority (NDMA) coordinates relief across 28 states. As AI-assisted decision support systems enter this space, **the bias they carry could cost lives**.

Research shows:
- Post-disaster resource allocation systematically underfunds areas with >60% BPL population
- Peri-urban and rural zones receive help 2.3× later than urban zones after the same disaster
- Current ML-based resource optimizers maximize throughput, not equity

FairRecovery++ is a proof-of-concept that **fairness can be trained into AI allocators** — not just as a constraint, but as a core objective the model genuinely learns to optimize.

---

## 🔬 Environment Design

The environment is OpenEnv-compliant with a Gym-style API:

```python
env = FairRecoveryEnvironment()
obs = env.reset(difficulty="hard", seed=42)

# obs contains:
# - zones: list of ZoneState (damage, vulnerable_ratio, service level)
# - budget_left: remaining resource budget
# - fairness_score: current equity index
# - step_stage: "analyze" | "allocate" | "execute"

action = FairRecoveryAction(
    action_type="analyze",
    critical_zones=[4, 3]
)
obs = env.step(action)
```

**Three-phase cycle per day:**
1. **Analyze** — identify critical zones (agent chooses which zones to prioritize)
2. **Allocate** — dispatch resources (medical teams, power repair, water supply)
3. **Execute** — environment updates zone states based on allocations

---

## 🚀 Try It Yourself

**Live Demo:** [HuggingFace Space →](https://huggingface.co/spaces/Joshua1702/FairRecovery-PlusPlus)

**Training Notebook:** [Open in GitHub →](https://github.com/joshua400/FairRecovery-PlusPlus/blob/main/train.ipynb)

**GitHub Repository:** [joshua400/FairRecovery-PlusPlus →](https://github.com/joshua400/FairRecovery-PlusPlus)

---

## 📖 What's Next

- [ ] Multi-agent version: competing AI planners across state boundaries  
- [ ] Real NDMA district data integration  
- [ ] Adversarial scenarios: corrupt allocation attempts, misinformation about zone status  
- [ ] Larger model (Llama 3.2 3B) training run  

---

## 🙏 Acknowledgements

Built for the **OpenEnv Hackathon India 2026**.  
Built with OpenEnv, HuggingFace TRL, Unsloth, and Gradio.

---

*If you've ever watched relief trucks drive past a slum to reach a gated colony first — this project is for you.*
