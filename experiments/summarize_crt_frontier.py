"""Summarize CRT frontier runs."""
import csv
import re
from pathlib import Path
import numpy as np

base = Path('results')
rows = []
for raw in base.glob('crt_frontier_lam*_eps*/raw.csv'):
    m = re.search(r'lam([0-9.]+)_eps([0-9.]+)', str(raw.parent))
    if not m:
        continue
    lam, eps = m.group(1), m.group(2)
    with raw.open(newline='', encoding='utf-8') as f:
        data = list(csv.DictReader(f))
    groups = {}
    for r in data:
        key = (r['ranks'], r['checkpoint'], r['budget'])
        groups.setdefault(key, {}).setdefault(r['method'], []).append(r)
    for key, by_method in groups.items():
        if 'standard' not in by_method or 'crt_sri' not in by_method:
            continue
        def mean(method, col):
            return float(np.mean([float(x[col]) for x in by_method[method]]))
        rows.append({
            'lambda': lam,
            'epsilon': eps,
            'ranks': key[0],
            'checkpoint': key[1],
            'budget': key[2],
            'clean_delta': mean('crt_sri','clean_reward') - mean('standard','clean_reward'),
            'gap_damage_delta': mean('crt_sri','gap_damage') - mean('standard','gap_damage'),
            'p1_delta': mean('crt_sri','p1') - mean('standard','p1'),
            'gini_delta': mean('crt_sri','dependence_gini') - mean('standard','dependence_gini'),
        })

outdir = base / 'crt_frontier_summary'
outdir.mkdir(parents=True, exist_ok=True)
with (outdir / 'raw.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
summary = []
for lam in sorted({r['lambda'] for r in rows}, key=float):
    for eps in sorted({r['epsilon'] for r in rows if r['lambda']==lam}, key=float):
        subset = [r for r in rows if r['lambda']==lam and r['epsilon']==eps]
        summary.append({
            'lambda': lam,
            'epsilon': eps,
            'mean_clean_delta': float(np.mean([r['clean_delta'] for r in subset])),
            'mean_gap_damage_delta': float(np.mean([r['gap_damage_delta'] for r in subset])),
            'mean_p1_delta': float(np.mean([r['p1_delta'] for r in subset])),
            'mean_gini_delta': float(np.mean([r['gini_delta'] for r in subset])),
            'n': len(subset),
        })
with (outdir / 'summary.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader(); w.writerows(summary)
with (outdir / 'summary.md').open('w', encoding='utf-8') as f:
    f.write('# CRT Frontier Summary\n\n')
    f.write('| lambda | epsilon | clean delta | damage delta | P1 delta | Gini delta | n |\n')
    f.write('|---:|---:|---:|---:|---:|---:|---:|\n')
    for r in summary:
        f.write(f"| {r['lambda']} | {r['epsilon']} | {r['mean_clean_delta']:+.3f} | {r['mean_gap_damage_delta']:+.3f} | {r['mean_p1_delta']:+.3f} | {r['mean_gini_delta']:+.3f} | {r['n']} |\n")
print(outdir)
