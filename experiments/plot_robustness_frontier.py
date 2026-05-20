"""Plot the robustness--exploitation frontier from the JSON dump."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    figdir = Path(__file__).resolve().parents[1] / "paper" / "latex" / "figures"
    with open(figdir / "robustness_frontier.json", "r", encoding="utf-8") as fh:
        data = json.load(fh)

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=False)
    for ax, game in zip(axes, ("kuhn", "leduc")):
        rows = data[game]
        cleans = [r["clean_mean"] for r in rows]
        attacks = [r["attacked_mean"] for r in rows]
        clean_ci = [r["clean_ci"] for r in rows]
        attack_ci = [r["attacked_ci"] for r in rows]
        lambdas = [r["lambda"] for r in rows]

        sc = ax.errorbar(
            cleans,
            attacks,
            xerr=clean_ci,
            yerr=attack_ci,
            fmt="o-",
            color="#2c7fb8",
            ecolor="#7fcdbb",
            elinewidth=1.0,
            capsize=2.0,
            markersize=5.0,
        )
        for x, y, lam in zip(cleans, attacks, lambdas):
            ax.annotate(
                rf"$\lambda{{=}}{lam:.3f}$",
                xy=(x, y),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
                color="#444",
            )
        ax.axhline(rows[0]["attacked_mean"], color="#bbb", linestyle="--", linewidth=0.8)
        ax.axvline(rows[0]["clean_mean"], color="#bbb", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Clean reward")
        ax.set_ylabel("Adversarial reward")
        ax.set_title(f"{game.title()}")
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        "Robustness--exploitation frontier: CACv-regularized self-play",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_pdf = figdir / "robustness_frontier.pdf"
    fig.savefig(out_pdf, bbox_inches="tight")
    print("Wrote", out_pdf)


if __name__ == "__main__":
    main()
