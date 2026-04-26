# 🏙️ FairRecovery++: Teaching LLMs to Make Fair Disaster Recovery Decisions

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

## 🎯 The Fairness Trap

The hard scenario is deliberately designed with a trap:
- **Zone 0** (wealthy district): 35% damage, easy to fix, 8% vulnerable population.
- **Zone 4** (informal settlement): 92% damage, 96% vulnerable population.

A naive utility-maximizing agent always picks Zone 0: lower cost, faster payoff, higher immediate reward. Zone 4 gets ignored.

Our trained **Qwen-7B** and **Llama-3.2-1B** agents learn to prioritize Zone 4 — because its 96% vulnerable population deserves equitable access to recovery services, even if it costs more per unit of service gained.

---

## 🧠 Training & Results

We trained two models using GRPO (TRL + Unsloth), comparing them against a greedy damage-only baseline:

| Metric | Greedy Baseline | Llama-3.2-1B | Qwen-2.5-7B-GRPO |
|--------|----------------|--------------|------------------|
| **Avg Episode Reward** | 0.548 | 0.785 | **0.864** (+57%) |
| **Avg Final Fairness** | 0.732 | 0.840 | **0.912** (+24%) |
| **Avg Final Utility**  | 0.545 | 0.712 | **0.808** (+48%) |
| **Strategy discovered** | Always Zone 0 | Vulnerable-first | Strategic-equitable |

### 📈 Visual Evidence

**Dual-Model Comparison**
![Model Comparison](evidence/plots/model_comparison.png)
*Figure 1: Comparison between Baseline, Llama-1B, and Qwen-7B.*

**Baseline vs Trained (Qwen)**
![Training Results](evidence/plots/training_results.png)
*Figure 2: 57% reward improvement.*

**Fairness-Utility Frontier**
![Utility vs Fairness](evidence/plots/utility_vs_fairness.png)
*Figure 3: Escaping the greed trap.*

The trained agents spontaneously discovered a "**medical-first equity**" strategy: deploy medical resources to the highest-vulnerability zones first, then return to efficiency-optimized zones. This is exactly the pattern that disaster recovery experts recommend — and the agents learned it from reward signals alone.

---

## 🌍 Why This Matters

Post-disaster resource allocation is a multi-billion dollar annual challenge. AI systems that optimize only for aggregate efficiency will systematically disadvantage already-vulnerable communities. FairRecovery++ provides a rigorous, reproducible benchmark for training and evaluating LLM agents that balance these competing objectives.

---

*FairRecovery++ was built for the Meta PyTorch OpenEnv Hackathon India 2026.*  
*Primary Theme: 3.1 (Real-World Professional Tasks) | Secondary Theme: 2 (Long-Horizon Planning)*
