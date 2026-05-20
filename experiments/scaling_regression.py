"""Scaling regression: log(states) vs adversary ratio.

Uses existing data points from all scale experiments.
Fits log-linear regression and reports R^2.
"""
import numpy as np

# Data from experiments (states, adv_reward, random_reward, none_reward)
# States = reachable P0 info-state count under uniform random play (game-tree
# size proxy). This is monotone in rank count, unlike trained-policy observed
# counts which depend on policy concentration.
data = [
    ("Kuhn",    6,      -0.95,  -0.22,  +0.01),
    ("Leduc",   144,    -2.57,  -1.17,  -0.10),
    ("Leduc-5", 390,    -2.74,  -0.60,  -0.15),
    ("Leduc-10", 1530,  -3.19,  -0.68,  -0.18),
    ("Leduc-20", 5900,  -3.00,  -0.63,  -0.09),
    ("Leduc-30", 12248, -3.218, -0.454, -0.081),
    ("Leduc-50", 26293, -2.965, -0.475, -0.098),
]

print("Scaling Analysis: Adversary Efficiency vs Game Size", flush=True)
print("="*60, flush=True)

names = [d[0] for d in data]
states = np.array([d[1] for d in data], dtype=float)
adv = np.array([d[2] for d in data])
rand = np.array([d[3] for d in data])
none = np.array([d[4] for d in data])

# Compute damage ratios
# Ratio = (none - adv) / (none - rand) = how much more damage adv does vs random
damage_adv = none - adv   # positive = more damage
damage_rand = none - rand  # positive = more damage
ratio = damage_adv / np.maximum(damage_rand, 0.01)

log_states = np.log10(states)

print(f"\n{'Game':>12s}  {'States':>6s}  {'log10':>6s}  {'Dmg_adv':>8s}  {'Dmg_rand':>8s}  {'Ratio':>6s}")
print("-"*55)
for i, name in enumerate(names):
    print(f"{name:>12s}  {states[i]:>6.0f}  {log_states[i]:>6.2f}  "
          f"{damage_adv[i]:>8.2f}  {damage_rand[i]:>8.2f}  {ratio[i]:>6.1f}x")

# Log-linear regression: ratio = a * log10(states) + b
A = np.vstack([log_states, np.ones(len(log_states))]).T
slope, intercept = np.linalg.lstsq(A, ratio, rcond=None)[0]

# R^2
predicted = slope * log_states + intercept
ss_res = np.sum((ratio - predicted)**2)
ss_tot = np.sum((ratio - np.mean(ratio))**2)
r_squared = 1 - ss_res / ss_tot

# Pearson correlation
pearson_r = np.corrcoef(log_states, ratio)[0, 1]

print(f"\nLog-linear fit: ratio = {slope:.2f} * log10(states) + {intercept:.2f}")
print(f"R^2 = {r_squared:.4f}")
print(f"Pearson r(log10(states), ratio) = {pearson_r:.4f}")
print(f"\nInterpretation: each 10x increase in state space -> {slope:.1f}x increase in adversary advantage ratio")

# Also compute: absolute damage gap (adv - rand)
gap = adv - rand
print(f"\nAbsolute damage gap (adv_reward - rand_reward):")
for i, name in enumerate(names):
    print(f"  {name:>12s}: {gap[i]:+.2f}")

# Excluding Kuhn (2-action saturation outlier)
print(f"\n--- Excluding Kuhn (2 actions, ceiling effect) ---")
idx = slice(1, None)
slope2, intercept2 = np.linalg.lstsq(
    np.vstack([log_states[idx], np.ones(len(log_states[idx]))]).T,
    ratio[idx], rcond=None)[0]
pred2 = slope2 * log_states[idx] + intercept2
ss_res2 = np.sum((ratio[idx] - pred2)**2)
ss_tot2 = np.sum((ratio[idx] - np.mean(ratio[idx]))**2)
r2_2 = 1 - ss_res2 / ss_tot2
print(f"Fit: ratio = {slope2:.2f} * log10(states) + {intercept2:.2f}")
print(f"R^2 = {r2_2:.4f}")
print(f"Pearson r = {np.corrcoef(log_states[idx], ratio[idx])[0,1]:.4f}")
