"""Aggregate strategic concentration profile runs into paper-ready tables/figures."""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def f(row, key):
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if not math.isnan(x) and not math.isnan(y)]
    if len(pairs) < 2:
        return float("nan")
    x, y = zip(*pairs)
    return float(np.corrcoef(x, y)[0, 1])


def correlations(rows):
    metrics = ["p1", "p2", "p4", "p8", "p16", "dependence_gini", "dependence_total", "entropy", "sri", "clean_reward"]
    targets = ["gap_damage", "random_damage"]
    scopes = [("all", rows)]
    for victim in sorted({r["victim"] for r in rows}):
        scopes.append((f"victim={victim}", [r for r in rows if r["victim"] == victim]))
    for method in sorted({r["method"] for r in rows}):
        scopes.append((f"method={method}", [r for r in rows if r["method"] == method]))
    for ranks in sorted({int(r["ranks"]) for r in rows}):
        scopes.append((f"leduc{ranks}", [r for r in rows if int(r["ranks"]) == ranks]))
    out = []
    for scope, subset in scopes:
        for metric in metrics:
            for target in targets:
                out.append({
                    "scope": scope,
                    "metric": metric,
                    "target": target,
                    "pearson": pearson([f(r, metric) for r in subset], [f(r, target) for r in subset]),
                    "n": len(subset),
                })
    return out


def intervention_summary(rows):
    q = [r for r in rows if r["victim"] == "q" and r["method"] in {"standard", "crt_sri", "entropy"}]
    keys = sorted({(r["ranks"], r["checkpoint"], r["budget"]) for r in q})
    out = []
    for ranks, checkpoint, budget in keys:
        group = [r for r in q if (r["ranks"], r["checkpoint"], r["budget"]) == (ranks, checkpoint, budget)]
        means = {}
        for method in ["standard", "entropy", "crt_sri"]:
            subset = [r for r in group if r["method"] == method]
            if not subset:
                continue
            means[method] = {
                "gap_damage": float(np.mean([f(r, "gap_damage") for r in subset])),
                "random_damage": float(np.mean([f(r, "random_damage") for r in subset])),
                "p1": float(np.mean([f(r, "p1") for r in subset])),
                "p4": float(np.mean([f(r, "p4") for r in subset])),
                "gini": float(np.mean([f(r, "dependence_gini") for r in subset])),
                "clean": float(np.mean([f(r, "clean_reward") for r in subset])),
            }
        if "standard" in means and "crt_sri" in means:
            out.append({
                "ranks": ranks,
                "checkpoint": checkpoint,
                "budget": budget,
                "standard_gap_damage": means["standard"]["gap_damage"],
                "crt_gap_damage": means["crt_sri"]["gap_damage"],
                "crt_gap_damage_delta": means["crt_sri"]["gap_damage"] - means["standard"]["gap_damage"],
                "standard_p1": means["standard"]["p1"],
                "crt_p1": means["crt_sri"]["p1"],
                "crt_p1_delta": means["crt_sri"]["p1"] - means["standard"]["p1"],
                "standard_gini": means["standard"]["gini"],
                "crt_gini": means["crt_sri"]["gini"],
                "crt_gini_delta": means["crt_sri"]["gini"] - means["standard"]["gini"],
                "standard_clean": means["standard"]["clean"],
                "crt_clean": means["crt_sri"]["clean"],
            })
    return out


def write_markdown(path, corr, intervention):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Strategic Concentration Strong-Accept Results", ""]
    lines += ["## Top Correlations", "", "| Scope | Metric | Target | r | n |", "|---|---|---|---:|---:|"]
    top = [r for r in corr if r["scope"] in {"all", "victim=dqn", "victim=q"} and r["target"] == "gap_damage"]
    order = {"p1": 0, "p2": 1, "p4": 2, "p8": 3, "p16": 4, "dependence_gini": 5, "entropy": 6, "sri": 7, "clean_reward": 8}
    top = sorted(top, key=lambda r: (r["scope"], order.get(r["metric"], 99)))
    for r in top:
        lines.append(f"| {r['scope']} | {r['metric']} | {r['target']} | {float(r['pearson']):.3f} | {r['n']} |")
    lines += ["", "## CRT Intervention Summary", "", "Mean deltas are CRT-SRI minus standard for tabular Q-learning.", "", "| Quantity | Mean Delta |", "|---|---:|"]
    if intervention:
        lines.append(f"| Gap damage | {np.mean([f(r, 'crt_gap_damage_delta') for r in intervention]):+.3f} |")
        lines.append(f"| P1 share | {np.mean([f(r, 'crt_p1_delta') for r in intervention]):+.3f} |")
        lines.append(f"| Gini | {np.mean([f(r, 'crt_gini_delta') for r in intervention]):+.3f} |")
        lines.append(f"| Clean reward | {np.mean([f(r, 'crt_clean') - f(r, 'standard_clean') for r in intervention]):+.3f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_scatter(outdir, rows):
    outdir.mkdir(parents=True, exist_ok=True)
    damage = np.array([f(r, "gap_damage") for r in rows])
    specs = [("p1", "P_pi(1)"), ("dependence_gini", "Strategic dependence Gini"), ("entropy", "Policy entropy"), ("sri", "SRI")]
    for key, label in specs:
        xs = np.array([f(r, key) for r in rows])
        plt.figure(figsize=(4.4, 3.4))
        for victim, marker in [("dqn", "o"), ("q", "x")]:
            idx = [i for i, r in enumerate(rows) if r["victim"] == victim]
            plt.scatter(xs[idx], damage[idx], s=16, alpha=0.65, marker=marker, label=victim)
        plt.xlabel(label)
        plt.ylabel("Gap-attack damage")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(str(outdir / f"{key}_vs_gap_damage.png"), dpi=220)
        plt.close()


def plot_lorenz(outdir, lorenz_paths):
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.2, 3.8))
    count = 0
    for path in lorenz_paths:
        rows = read_csv(path)
        labels = []
        for row in rows:
            if row["label"] not in labels:
                labels.append(row["label"])
        for label in labels[:6]:
            series = [r for r in rows if r["label"] == label]
            xs = [f(r, "rank_fraction") for r in series]
            ys = [f(r, "mass_fraction") for r in series]
            plt.plot(xs, ys, alpha=0.6, linewidth=1.2)
            count += 1
            if count >= 12:
                break
        if count >= 12:
            break
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.xlabel("Fraction of reached states")
    plt.ylabel("Cumulative dependence mass")
    plt.title("Strategic Lorenz curves")
    plt.tight_layout()
    plt.savefig(str(outdir / "lorenz_curves.png"), dpi=220)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", default=[
        "results/strategic_concentration_dqn_scaling/raw.csv",
        "results/strategic_concentration_q_crt_scaling/raw.csv",
    ])
    parser.add_argument("--lorenz", nargs="+", default=[
        "results/strategic_concentration_dqn_scaling/lorenz_curves.csv",
        "results/strategic_concentration_q_crt_scaling/lorenz_curves.csv",
    ])
    parser.add_argument("--outdir", type=Path, default=Path("results/strategic_concentration_combined"))
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        rows.extend(read_csv(path))
    corr = correlations(rows)
    intervention = intervention_summary(rows)
    write_csv(args.outdir / "raw.csv", rows)
    write_csv(args.outdir / "correlations.csv", corr)
    write_csv(args.outdir / "crt_intervention.csv", intervention)
    write_markdown(args.outdir / "summary.md", corr, intervention)
    plot_scatter(args.outdir / "figures", rows)
    plot_lorenz(args.outdir / "figures", args.lorenz)
    print(f"Wrote combined strategic concentration summary to {args.outdir}")


if __name__ == "__main__":
    main()

