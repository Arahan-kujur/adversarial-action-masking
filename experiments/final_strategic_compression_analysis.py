"""Final strategic-compression analysis and figures.

Consumes outputs from strategic_concentration_profiles.py and produces paper-ready
summary tables/figures for scaling, entropy failure, longitudinal curves, Lorenz
curves, and CRT intervention frontiers.
"""

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def val(row, key):
    try:
        return float(row[key])
    except Exception:
        return float("nan")


def pearson(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if not math.isnan(float(x)) and not math.isnan(float(y))]
    if len(pairs) < 2:
        return float("nan")
    x, y = zip(*pairs)
    return float(np.corrcoef(x, y)[0, 1])


def grouped(rows, keys):
    out = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        out.setdefault(key, []).append(row)
    return out


def correlations(rows):
    metrics = ["p1", "p2", "p4", "p8", "p16", "dependence_gini", "dependence_total", "entropy", "sri", "clean_reward"]
    targets = ["gap_damage", "random_damage"]
    scopes = [("all", rows)]
    for victim in sorted({r.get("victim", "") for r in rows}):
        scopes.append((f"victim={victim}", [r for r in rows if r.get("victim") == victim]))
    for method in sorted({r.get("method", "") for r in rows}):
        scopes.append((f"method={method}", [r for r in rows if r.get("method") == method]))
    for ranks in sorted({r.get("ranks", "") for r in rows}, key=lambda x: int(x) if str(x).isdigit() else 0):
        scopes.append((f"leduc{ranks}", [r for r in rows if r.get("ranks") == ranks]))
    out = []
    for scope, subset in scopes:
        if len(subset) < 2:
            continue
        for metric in metrics:
            for target in targets:
                out.append({
                    "scope": scope,
                    "metric": metric,
                    "target": target,
                    "pearson": pearson([val(r, metric) for r in subset], [val(r, target) for r in subset]),
                    "n": len(subset),
                })
    return out


def longitudinal_summary(rows):
    out = []
    for key, subset in grouped(rows, ["victim", "method", "ranks", "checkpoint"]).items():
        victim, method, ranks, checkpoint = key
        out.append({
            "victim": victim,
            "method": method,
            "ranks": ranks,
            "checkpoint": checkpoint,
            "n": len(subset),
            "clean_reward": np.mean([val(r, "clean_reward") for r in subset]),
            "gap_damage": np.mean([val(r, "gap_damage") for r in subset]),
            "random_damage": np.mean([val(r, "random_damage") for r in subset]),
            "p1": np.mean([val(r, "p1") for r in subset]),
            "p4": np.mean([val(r, "p4") for r in subset]),
            "p16": np.mean([val(r, "p16") for r in subset]),
            "gini": np.mean([val(r, "dependence_gini") for r in subset]),
            "entropy": np.mean([val(r, "entropy") for r in subset]),
            "sri": np.mean([val(r, "sri") for r in subset]),
        })
    return sorted(out, key=lambda r: (r["victim"], r["method"], int(r["ranks"]), int(r["checkpoint"])))


def intervention_summary(rows):
    q = [r for r in rows if r.get("victim") == "q" and r.get("method") in {"standard", "entropy", "crt_sri"}]
    out = []
    for key, subset in grouped(q, ["ranks", "checkpoint", "budget"]).items():
        ranks, checkpoint, budget = key
        by_method = grouped(subset, ["method"])
        if ("standard",) not in by_method or ("crt_sri",) not in by_method:
            continue
        std = by_method[("standard",)]
        crt = by_method[("crt_sri",)]
        ent = by_method.get(("entropy",), [])
        def mean(rows_, key_): return np.mean([val(r, key_) for r in rows_]) if rows_ else float("nan")
        out.append({
            "ranks": ranks,
            "checkpoint": checkpoint,
            "budget": budget,
            "standard_damage": mean(std, "gap_damage"),
            "crt_damage": mean(crt, "gap_damage"),
            "entropy_damage": mean(ent, "gap_damage"),
            "crt_damage_delta": mean(crt, "gap_damage") - mean(std, "gap_damage"),
            "crt_p1_delta": mean(crt, "p1") - mean(std, "p1"),
            "crt_gini_delta": mean(crt, "dependence_gini") - mean(std, "dependence_gini"),
            "crt_clean_delta": mean(crt, "clean_reward") - mean(std, "clean_reward"),
        })
    return out


def plot_metric_vs_damage(rows, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    specs = [("p1", "Top-1 dependence share"), ("p16", "Top-16 dependence share"), ("dependence_gini", "Dependence Gini"), ("entropy", "Policy entropy"), ("sri", "SRI")]
    damage = np.array([val(r, "gap_damage") for r in rows])
    for metric, label in specs:
        xs = np.array([val(r, metric) for r in rows])
        plt.figure(figsize=(4.6, 3.5))
        for method, marker in [("standard", "o"), ("entropy", "^"), ("crt_sri", "x")]:
            idx = [i for i, r in enumerate(rows) if r.get("method") == method]
            if idx:
                plt.scatter(xs[idx], damage[idx], s=18, alpha=0.65, marker=marker, label=method)
        plt.xlabel(label)
        plt.ylabel("Gap-attack damage")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(outdir / f"{metric}_vs_gap_damage.png", dpi=240)
        plt.close()


def plot_longitudinal(long_rows, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    for ranks in sorted({r["ranks"] for r in long_rows}, key=int):
        subset = [r for r in long_rows if r["victim"] == "dqn" and r["method"] == "standard" and r["ranks"] == ranks]
        if not subset:
            continue
        xs = np.array([int(r["checkpoint"]) for r in subset])
        order = np.argsort(xs)
        xs = xs[order]
        plt.figure(figsize=(5.0, 3.5))
        for metric, label in [("clean_reward", "Clean reward"), ("gap_damage", "Gap damage"), ("p16", "P_pi(16)"), ("entropy", "Entropy")]:
            ys = np.array([float(r[metric]) for r in subset])[order]
            if np.nanmax(ys) > np.nanmin(ys):
                ys = (ys - np.nanmin(ys)) / (np.nanmax(ys) - np.nanmin(ys))
            plt.plot(xs, ys, marker="o", label=label)
        plt.xscale("log")
        plt.xlabel("Training episodes")
        plt.ylabel("Normalized value")
        plt.title(f"Leduc-{ranks} DQN longitudinal compression")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(outdir / f"longitudinal_leduc{ranks}.png", dpi=240)
        plt.close()


def plot_lorenz(lorenz_paths, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.2, 3.8))
    count = 0
    for path in lorenz_paths:
        rows = read_csv(path)
        labels = []
        for row in rows:
            if row["label"] not in labels:
                labels.append(row["label"])
        for label in labels[:8]:
            series = [r for r in rows if r["label"] == label]
            xs = [val(r, "rank_fraction") for r in series]
            ys = [val(r, "mass_fraction") for r in series]
            plt.plot(xs, ys, alpha=0.65, linewidth=1.2)
            count += 1
            if count >= 16:
                break
        if count >= 16:
            break
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.xlabel("Fraction of reached states")
    plt.ylabel("Cumulative dependence mass")
    plt.title("Strategic Lorenz curves")
    plt.tight_layout()
    plt.savefig(outdir / "lorenz_curves.png", dpi=240)
    plt.close()


def markdown_summary(path, corr, intervention):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Final Strategic Compression Summary", "", "## Correlations", "", "| Scope | Metric | Target | r | n |", "|---|---|---|---:|---:|"]
    for row in corr:
        if row["scope"] in {"all", "victim=dqn", "victim=q"} and row["target"] == "gap_damage" and row["metric"] in {"p1", "p4", "p16", "dependence_gini", "entropy", "sri", "clean_reward"}:
            lines.append(f"| {row['scope']} | {row['metric']} | {row['target']} | {float(row['pearson']):.3f} | {row['n']} |")
    lines += ["", "## CRT Intervention", "", "| Quantity | Mean delta |", "|---|---:|"]
    if intervention:
        lines.append(f"| Gap damage | {np.mean([val(r, 'crt_damage_delta') for r in intervention]):+.3f} |")
        lines.append(f"| P1 share | {np.mean([val(r, 'crt_p1_delta') for r in intervention]):+.3f} |")
        lines.append(f"| Gini | {np.mean([val(r, 'crt_gini_delta') for r in intervention]):+.3f} |")
        lines.append(f"| Clean reward | {np.mean([val(r, 'crt_clean_delta') for r in intervention]):+.3f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--lorenz", nargs="+", default=[])
    parser.add_argument("--outdir", type=Path, default=Path("results/strategic_compression_final"))
    args = parser.parse_args()
    rows = []
    for input_path in args.inputs:
        rows.extend(read_csv(input_path))
    corr = correlations(rows)
    long_rows = longitudinal_summary(rows)
    intervention = intervention_summary(rows)
    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(args.outdir / "raw.csv", rows)
    write_csv(args.outdir / "correlations.csv", corr)
    write_csv(args.outdir / "longitudinal.csv", long_rows)
    write_csv(args.outdir / "crt_intervention.csv", intervention)
    markdown_summary(args.outdir / "summary.md", corr, intervention)
    plot_metric_vs_damage(rows, args.outdir / "figures")
    plot_longitudinal(long_rows, args.outdir / "figures")
    plot_lorenz(args.lorenz, args.outdir / "figures")
    print(f"Wrote final strategic compression analysis to {args.outdir}")


if __name__ == "__main__":
    main()
