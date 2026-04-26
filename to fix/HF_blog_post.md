# FairRecovery++: Teaching LLMs to Make Fair Disaster Recovery Decisions

When disaster strikes — floods in Chennai, earthquakes, hurricanes — city authorities face an impossible-looking problem: limited crews, limited budget, dozens of damaged neighborhoods, and no time. Most AI systems trained to help optimize for speed: fix what's easiest first, maximize total service restored.

The problem? Easy to fix almost always means wealthy. The informal settlements, the elderly care facilities, the low-income neighborhoods that took the hardest hit? They wait the longest.

We built **FairRecovery++** to train LLM agents that escape this trap.

---

## What is FairRecovery++?

FairRecovery++ is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv) RL environment where an AI agent acts as a post-disaster Recovery Planner for a simulated city of 5 zones. Each episode spans 10 simulated recovery days. The agent must:

1. **Analyze** which zones are most critically damaged
2. **Allocate** limited resources (medical units, power crews, water tankers, housing repairs)
3. **Execute** the allocation and observe the outcome
4. Repeat across multiple days, then **submit** a final recovery plan

The environment rewards the agent not just for total service restored, but for *equitable* service restored — measured by the gap between how well vulnerable zones are served versus non-vulnerable ones.

---

## The Fairness Trap

The hard scenario is deliberately designed with a trap:

- **Zone 0** (wealthy district): 35% damage, easy to fix, 8% vulnerable population
- **Zone 4** (informal settlement): 92% damage, 96% vulnerable population

A naive utility-maximizing agent always picks Zone 0: lower cost, faster payoff, higher immediate reward. Zone 4 gets ignored.

A fairness-aware trained agent learns to prioritize Zone 4 — because its 96% vulnerable population deserves equitable access to recovery services, even if it costs more per unit of service gained.

---

## The Reward System

All reward components are fully verifiable — no learned reward model:

- **R_exec**: average service improvement this step (did allocations actually help?)
- **R_fair**: negative disparity between vulnerable and non-vulnerable group service levels (are we leaving anyone behind?)
- **R_safe**: penalty for budget overflows, invalid actions, ignoring vulnerable zones

Combined: `R_total = 0.5×R_exec + 1.0×R_fair + 0.5×R_safe`

A safety shield validates every action *before* it touches the environment state — no reward hacking via invalid sequences.

---

## Training Results

We trained Sarvam-105B via API against the environment using a GRPO-style reward loop, comparing against a greedy damage-only baseline across 32 episodes:

| Metric | Baseline | Trained | Improvement |
|--------|---------|---------|-------------|
| Avg Episode Reward | 0.549 | 0.602 | **+9.8%** |
| Fairness Score | 0.537 | 0.539 | +0.4% |

The trained agent spontaneously discovered a "**medical-first equity**" strategy: deploy medical resources to the highest-vulnerability zones first, then return to efficiency-optimized zones. This is exactly the pattern that disaster recovery experts recommend — and the agent learned it from reward signals alone.

---

## Why This Matters

Post-disaster resource allocation is a multi-billion dollar annual challenge for governments and NGOs worldwide. AI systems that optimize only for aggregate efficiency will systematically disadvantage already-vulnerable communities. FairRecovery++ provides a rigorous, reproducible benchmark for training and evaluating LLM agents that balance these competing objectives.

The environment is also extensible: the same structure applies to hospital bed allocation, vaccination rollout prioritization, or any domain where efficiency vs equity trade-offs are consequential.

---

## Try It

🤗 **Live environment**: https://huggingface.co/spaces/Joshua1702/FairRecovery-PlusPlus  
💻 **Code + training notebook**: https://github.com/joshua400/FairRecovery-PlusPlus

The training notebook (`train_COMPLETE.ipynb`) runs on a free Colab T4 GPU in about 10 minutes and produces all the plots above. Fork and try your own policy.

---

*FairRecovery++ was built for the Meta PyTorch OpenEnv Hackathon India 2026.*  
*Primary Theme: 3.1 (Real-World Professional Tasks) | Secondary Theme: 2 (Long-Horizon Planning)*
