---
title: FairRecovery++
emoji: 🏗️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# FairRecovery: Training an LLM to Make Fair Disaster Recovery Decisions

After a flood, city authorities must decide: which neighborhood gets power restored first? Which gets medical units? The obvious answer — fix the easiest zones first — consistently leaves the most vulnerable people waiting the longest.

We built **FairRecovery**, an OpenEnv RL environment where an LLM agent must allocate scarce resources (power, water, medical) across 5 city zones over 10 days, under budget constraints, while keeping vulnerable populations from falling behind.

![FairRecovery++ Dashboard](https://raw.githubusercontent.com/joshua400/FairRecovery-PlusPlus/main/assets/dashboard.png)

## The Fairness Trap
The environment features a mathematical **Fairness Trap**: Zone 4 is the most damaged (damage=0.92) AND the most vulnerable (96% vulnerable population). A naive, greedy agent optimizes Zone 0 (easiest to fix) and completely ignores Zone 4 to maximize raw utility. A trained agent learns to prioritize correctly.

### 📈 Learning to Prioritize
<p align="center">
  <img src="https://raw.githubusercontent.com/joshua400/FairRecovery-PlusPlus/main/assets/reward_vs_episode.png" width="45%" />
  <img src="https://raw.githubusercontent.com/joshua400/FairRecovery-PlusPlus/main/assets/fairness_vs_episode.png" width="45%" />
</p>
<p align="center"><i>Left: Reward growth over training | Right: Fairness index stabilization</i></p>

## Training the Model
We trained **Llama-3.2-1B-Instruct** with **GRPO** (via TRL + Unsloth) and measured improvement in both episode reward and fairness score before vs after training. The environment runs a 5-component composite rubric to prevent reward hacking.

- **Model:** Llama-3.2-1B-Instruct (4-bit quantized via Unsloth)
- **Method:** Group Relative Policy Optimization (GRPO)
- **Reward:** 0.4×Utility + 0.4×Fairness + 0.2×Safety

## 📊 Results

| Metric | Greedy Baseline | Trained LLM | Improvement |
|--------|----------------|-------------|-------------|
| Overall Reward | 0.781 | **0.814** | +4.2% |
| **Fairness (Equity Index)** | 0.837 | **0.854** | **+2.0%** |
| Utility | 0.552 | 0.561 | +1.6% |

## Try it yourself:
- **Try the environment UI:** [HuggingFace Space →](https://huggingface.co/spaces/Joshua1702/FairRecovery-PlusPlus)
- **Reproduce training in < 10 min:** [Google Colab →](https://github.com/joshua400/FairRecovery-PlusPlus/blob/main/train.ipynb)

---

## 🇮🇳 Why This Matters for India
India's National Disaster Management Authority (NDMA) coordinates relief across 28 states. As AI-assisted decision support systems enter this space, the bias they carry could cost lives. FairRecovery is a proof-of-concept that fairness can be trained into AI allocators — not just as a constraint, but as a core objective the model genuinely learns to optimize.

---

## 🙏 Acknowledgements
Built for the **OpenEnv Hackathon India 2026**.  
Built with OpenEnv, HuggingFace TRL, Unsloth, and Gradio.
