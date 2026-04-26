"""Generate publication-quality summary plots from episode_log.csv.

Reads:  episode_log.csv  (produced by train_sarvam_online.py)
Writes: assets/training_results.png  — bar chart Baseline vs Trained
        assets/score_heatmap.png     — per-episode reward heatmap
        assets/training_loss.png     — moving-avg curriculum reward over episodes
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

CSV = Path("episode_log.csv")
ASSETS = Path("assets")


def _load_csv(path: Path) -> Dict[str, List[Dict[str, float]]]:
    rows: Dict[str, List[Dict[str, float]]] = {"baseline": [], "trained": []}
    with path.open(encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for r in reader:
            rows[r["policy"]].append(
                {
                    "episode": int(r["episode"]),
                    "total_reward": float(r["total_reward"]),
                    "final_fairness": float(r["final_fairness"]),
                    "final_utility": float(r["final_utility"]),
                    "steps": int(r["steps"]),
                    "early_submits_blocked": int(r["early_submits_blocked"]),
                }
            )
    return rows


def training_results_bar(rows: Dict[str, List[Dict]], out: Path) -> None:
    metrics = {
        "Avg Reward (curriculum)": (
            np.mean([r["total_reward"] for r in rows["baseline"]]),
            np.mean([r["total_reward"] for r in rows["trained"]]),
        ),
        "Avg Final Utility": (
            np.mean([r["final_utility"] for r in rows["baseline"]]),
            np.mean([r["final_utility"] for r in rows["trained"]]),
        ),
        "Avg Final Fairness (0-1)": (
            np.mean([(r["final_fairness"] + 1) / 2 for r in rows["baseline"]]),
            np.mean([(r["final_fairness"] + 1) / 2 for r in rows["trained"]]),
        ),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = list(metrics.keys())
    base_vals = [metrics[k][0] for k in labels]
    trained_vals = [metrics[k][1] for k in labels]

    x = np.arange(len(labels))
    width = 0.35
    axes[0].bar(x - width / 2, base_vals, width, label="Baseline", color="#9aa0a6")
    axes[0].bar(x + width / 2, trained_vals, width, label="Trained (Qwen-7B-GRPO)", color="#1a73e8")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=15, ha="right")
    axes[0].set_ylabel("Score (0-1)")
    axes[0].set_title("Per-Metric: Baseline vs Trained (32 episodes)")
    axes[0].set_ylim(0, 1)
    axes[0].legend()
    axes[0].grid(alpha=0.25, axis="y")
    for i, (b, t) in enumerate(zip(base_vals, trained_vals)):
        axes[0].text(i - width / 2, b + 0.01, f"{b:.3f}", ha="center", fontsize=9)
        axes[0].text(i + width / 2, t + 0.01, f"{t:.3f}", ha="center", fontsize=9)

    delta_reward = float(np.mean([r["total_reward"] for r in rows["trained"]])) - float(
        np.mean([r["total_reward"] for r in rows["baseline"]])
    )
    delta_fair = float(
        np.mean([(r["final_fairness"] + 1) / 2 for r in rows["trained"]])
    ) - float(np.mean([(r["final_fairness"] + 1) / 2 for r in rows["baseline"]]))
    delta_util = float(
        np.mean([r["final_utility"] for r in rows["trained"]])
    ) - float(np.mean([r["final_utility"] for r in rows["baseline"]]))

    axes[1].bar(["Reward", "Utility", "Fairness"],
                [delta_reward, delta_util, delta_fair],
                color=["#1a73e8", "#34a853", "#fbbc04"])
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set_ylabel("Δ (Trained − Baseline)")
    axes[1].set_title("Improvement After Curriculum Training")
    axes[1].grid(alpha=0.25, axis="y")
    for i, v in enumerate([delta_reward, delta_util, delta_fair]):
        axes[1].text(i, v + (0.002 if v >= 0 else -0.005),
                     f"{v:+.3f}", ha="center",
                     va="bottom" if v >= 0 else "top", fontsize=10, fontweight="bold")

    fig.suptitle("FairRecovery++ — GRPO/RLVR-Style Training Results", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def score_heatmap(rows: Dict[str, List[Dict]], out: Path) -> None:
    base = [r["total_reward"] for r in rows["baseline"]]
    trained = [r["total_reward"] for r in rows["trained"]]
    n = max(len(base), len(trained))
    base += [np.nan] * (n - len(base))
    trained += [np.nan] * (n - len(trained))
    matrix = np.array([base, trained])

    fig, ax = plt.subplots(figsize=(14, 2.8))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.40, vmax=0.65)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Baseline", "Trained"])
    ax.set_xticks(np.arange(0, n, 2))
    ax.set_xticklabels(np.arange(1, n + 1, 2))
    ax.set_xlabel("Episode")
    ax.set_title("Per-Episode Reward Heatmap (32 baseline vs 32 trained)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Reward (0-1)")

    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def training_loss(rows: Dict[str, List[Dict]], out: Path) -> None:
    base = np.array([r["total_reward"] for r in rows["baseline"]])
    trained = np.array([r["total_reward"] for r in rows["trained"]])

    def _ma(arr: np.ndarray, w: int = 4) -> np.ndarray:
        if len(arr) == 0:
            return arr
        out_arr = np.zeros_like(arr, dtype=float)
        for i in range(len(arr)):
            s = max(0, i - w + 1)
            out_arr[i] = arr[s : i + 1].mean()
        return out_arr

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(range(1, len(base) + 1), _ma(base), label="Baseline (greedy heuristic)", color="#9aa0a6", linewidth=2)
    ax.plot(range(1, len(trained) + 1), _ma(trained), label="Trained (Qwen-7B-GRPO)", color="#1a73e8", linewidth=2)
    ax.fill_between(range(1, len(trained) + 1), _ma(trained), _ma(base), alpha=0.15, color="#1a73e8")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Curriculum Reward (4-ep moving avg)")
    ax.set_title("Curriculum Reward Curve — Trained Improves Over Baseline")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    ax.set_ylim(0.50, 0.65)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    rows = _load_csv(CSV)
    print(f"Baseline episodes: {len(rows['baseline'])}, Trained episodes: {len(rows['trained'])}")
    training_results_bar(rows, ASSETS / "training_results.png")
    score_heatmap(rows, ASSETS / "score_heatmap.png")
    training_loss(rows, ASSETS / "training_loss.png")
    print("Wrote:")
    print(f"  {ASSETS/'training_results.png'}")
    print(f"  {ASSETS/'score_heatmap.png'}")
    print(f"  {ASSETS/'training_loss.png'}")


if __name__ == "__main__":
    main()
