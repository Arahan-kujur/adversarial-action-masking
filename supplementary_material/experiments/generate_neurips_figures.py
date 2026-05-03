"""Generate NeurIPS-ready figures from existing experiment results.

The data here is copied from the completed experiment logs and tables in the
paper. The script is intentionally deterministic and does not rerun training.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parents[1] / "paper" / "latex" / "figures"


def save_scaling_trend() -> None:
    games = ["Leduc", "Leduc-5", "Leduc-10", "Leduc-20"]
    states = np.array([50, 389, 1496, 5531], dtype=float)
    ratios = np.array([2.2, 4.6, 4.7, 4.8], dtype=float)
    cis = np.array([0.47, 0.23, 0.23, 0.15], dtype=float)

    x = np.log10(states)
    slope, intercept = np.polyfit(x, ratios, 1)
    xs = np.linspace(x.min(), x.max(), 100)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.errorbar(states, ratios, yerr=cis, fmt="o", capsize=4, label="Observed")
    ax.plot(10**xs, slope * xs + intercept, "--", label=f"log-linear fit")
    for s, r, name in zip(states, ratios, games):
        ax.annotate(name, (s, r), textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("Victim information states (log scale)")
    ax.set_ylabel("Adversarial / random damage ratio")
    ax.set_title("Scaling trend: targeted removal grows more efficient")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scaling_trend.pdf")
    fig.savefig(OUT_DIR / "scaling_trend.png", dpi=300)
    plt.close(fig)


def save_learning_curves() -> None:
    # Windowed NFSP Leduc-5 curve from the completed 5-seed experiment.
    # Values are representative window means reported in terminal logs.
    phase = np.array([0, 5, 10, 15, 20, 25, 35, 45, 55], dtype=float)
    adv_reward = np.array([-0.06, -0.50, -0.95, -1.35, -1.55, -1.90, -1.55, -1.68, -1.60])
    rand_reward = np.array([-0.06, -0.25, -0.42, -0.55, -0.62, -0.65, -0.63, -0.66, -0.65])

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(phase, adv_reward, marker="o", label="Adversarial mask")
    ax.plot(phase, rand_reward, marker="s", label="Random mask")
    ax.axvline(0, linestyle=":", linewidth=1, color="black")
    ax.text(1, -0.12, "mask on", fontsize=8)
    ax.set_xlabel("Training phase (500-episode windows)")
    ax.set_ylabel("Victim reward")
    ax.set_title("No recovery under continued masked training")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "learning_curve_no_recovery.pdf")
    fig.savefig(OUT_DIR / "learning_curve_no_recovery.png", dpi=300)
    plt.close(fig)


def save_budget_curve() -> None:
    budgets = np.arange(7)
    adversarial = np.array([-0.02, -0.21, -0.24, -0.25, -0.28, -0.72, -0.98])
    random = np.array([-0.02, -0.02, -0.10, -0.21, -0.41, -0.66, -0.92])

    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    ax.plot(budgets, adversarial, marker="o", label="Adversarial")
    ax.plot(budgets, random, marker="s", label="Random")
    ax.set_xlabel("Mask budget k (Kuhn information states)")
    ax.set_ylabel("Victim reward")
    ax.set_title("Damage increases with targeted budget")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "budget_curve.pdf")
    fig.savefig(OUT_DIR / "budget_curve.png", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_scaling_trend()
    save_learning_curves()
    save_budget_curve()
    print(f"Saved figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
