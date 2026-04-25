# 🏗️ FairRecovery++: Escaping the Fairness Trap with Fair-GRPO-RLVR

[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/Joshua1702/FairRecovery-PlusPlus)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/joshua400/FairRecovery-PlusPlus/blob/main/train.ipynb)

**FairRecovery++** is a research-level reinforcement learning environment built for the **Meta OpenEnv Hackathon**. It challenges AI agents to manage disaster recovery not just efficiently, but **equitably**.

![Dashboard Screenshot](assets/dashboard.png)

---

## 🚨 The Challenge: The "Fairness Trap"

In disaster recovery, optimizing for "Efficiency" (overall service restored) often leads to a **Fairness Trap**. Naive AI agents naturally prioritise wealthy, low-damage zones because they offer "easy wins." Meanwhile, highly vulnerable, severely damaged zones are left behind, widening systemic inequality.

**FairRecovery++ forces agents to escape this trap by balancing efficiency vs. equity under tight budgets and adversarial pressure.**

---

## 🧠 Core Innovation: Fair-GRPO-RLVR

We introduce **Fair-GRPO-RLVR**, a multi-objective reinforcement learning framework that leverages:

1.  **Group Relative Policy Optimization (GRPO)**: An efficient, multi-sample policy gradient method tailored for complex decision-making.
2.  **Verifiable Reward Signals (RLVR)**: Transparent, formula-based rewards that eliminate "reward model hacking."
3.  **Inverse Service Disparity Index**: A novel fairness metric that penalizes the variance and gap between the most and least recovered zones.

### The Reward Formula
$$R_{total} = 0.4 \cdot \text{Utility} + 0.4 \cdot \text{Fairness (Equity)} + 0.2 \cdot \text{Safety}$$

---

## 📊 Proof of Learning

Our training results show that while a **Greedy Baseline** consistently fails the most vulnerable zones, the **Fair-GRPO-RLVR** agent learns to bridge the equity gap without sacrificing recovery speed.

| Metric | Baseline (Greedy) | **Trained (Fair-GRPO-RLVR)** | Improvement |
|--------|-------------------|-----------------------------|-------------|
| **Total Reward** | ~0.15 | **~0.85** | **+460%** |
| **Fairness (Equity)** | 0.32 | **0.94** | **+193%** |

### Multi-Objective Improvement
![Reward vs Episode](assets/reward_vs_episode.png)

### Closing the Equity Gap
![Fairness Improvement](assets/fairness_vs_episode.png)

---

## 🏗️ Architecture & Themes

### #1 Multi-Agent Interactions
The environment is alive. **Citizens** generate protests if ignored, **NGOs** deliver uncoordinated aid, and **Adversaries** target weak zones to disrupt the recovery. The agent must navigate these dynamics in real-time.

### #2 Long-Horizon Planning
Recovery happens over a multi-day protocol: `Analyze -> Allocate -> Execute -> Adapt`. Agents must plan budget across days, anticipating that early neglect of vulnerable zones leads to higher costs (and penalties) later.

### #3 World Modeling
A high-fidelity simulation where zone damage, service levels, and citizen satisfaction evolve dynamically. A **Safety Shield** ensures every action follows structural constraints and budget limits.

---

## 🚀 Getting Started

### 1. Interactive Demo (HF Spaces)
Experience the environment in your browser: [FairRecovery++ Space](https://huggingface.co/spaces/Joshua1702/FairRecovery-PlusPlus)

### 2. Training (Google Colab)
Train the **Llama-3.2-1B** model using Unsloth and GRPO in under 15 minutes: [Training Notebook](https://colab.research.google.com/github/joshua400/FairRecovery-PlusPlus/blob/main/train.ipynb)

---

## 🏆 Key Insight
**Optimizing for fairness is not just a moral choice—it improves long-term recovery efficiency.** By preventing "social collapse" in vulnerable zones, Fair-GRPO-RLVR ensures a stable, sustainable recovery for the entire city.

---

### 📝 Submission Details
- **Project**: FairRecovery++
- **Method**: Fair-GRPO-RLVR
- **Framework**: Meta OpenEnv / TRL / Unsloth
- **Authors**: Joshua Ragiland (Joshua1702)
