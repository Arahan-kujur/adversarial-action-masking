# Project Report

A narrative summary of the project, focusing on what was built, what was tried, what worked, and what failed. The README is the public-facing entry point; this document is the longer story for anyone who wants to understand how the work evolved.

## 1. Phenomenon and framing

A self-play agent in a discrete-action game can be hurt in several distinct ways:

- observation perturbations (the agent sees a noisy view of state),
- reward poisoning (the agent's learning signal is corrupted),
- action perturbations (the agent's chosen action is replaced by another *legal* action), or
- **action availability constraints** (the agent's set of *legal* actions is reduced at decision time, before action selection).

Most prior robust-RL work targets the first three. We targeted the fourth, motivated by realistic MAS settings where capabilities can be disabled by faults, sandboxing, rate limits, regulatory restrictions, or — in adversarial settings — by an attacker controlling the action interface. The structural difference matters: bounded perturbations preserve the agent's ability to choose; availability constraints can collapse a decision point to a singleton, eliminating contingent flexibility.

We frame the question quantitatively via **contingent action capacity**:

- `CAC_w(M) = Σ_h ρ(h) · 1[|M(h)| > 1]` — reach-weighted count of states that still admit a choice,
- `CAC_v(M) = Σ_h ρ(h) · δ(h) · 1[|M(h)| > 1]` — same, weighted by the local value gap `δ(h) = Q(h,a*) − Q(h,a₂)`.

`CAC_v` turns out to be both diagnostic and prescriptive: it explains *which* states matter (Section 3), it lower-bounds worst-case damage by the agent's top-`k` `ρ·δ` mass (Prop. 4), and the same quantity can be regularised during training to defend (Section 5.7).

## 2. Methodology

- **Self-play loop**: both players share a single learning agent; the constraint is applied to Player 0 during both training and evaluation.
- **Worst-case constraint as stress test**: we treat the constraint itself as a learned policy and train it in a bi-level loop (inner: 500 self-play episodes; outer: one REINFORCE step on the constraint policy with reward signal `−V_0`). We do this purely to upper-bound how badly a budget-`k` availability pattern can hurt a given agent.
- **Constraint parameterisations**: tabular preference table (with optional top-`k` confidence projection for budget enforcement), and an MLP that takes the agent's encoded info-state and outputs a distribution over `|A| + 1` choices (one per action, plus "no removal").
- **Agents**: tabular Q-learning, tabular PPO, tabular NFSP, neural NFSP, and DQN (with encoders for Kuhn, Leduc, and arbitrary Leduc-N).
- **Environments**: Kuhn poker, Leduc, Leduc-N (`N ∈ {5, 10, 20, 30, 50}`), a 5×5 competitive gridworld, a 4×4 resource-collection game, and two Hanabi variants (Hanabi-Small and a slightly larger Hanabi-V2 with proper rank distribution).

All environments are implemented from scratch (no OpenSpiel dependency) so that an availability mask can be inserted at every `legal_actions()` query and Leduc can be parameterised by rank count.

## 3. Scaling result

The headline empirical finding is a clean log-linear scaling trend across six Leduc-N sizes. We measure information-state count under uniform random play (a game-tree size proxy that is monotone in rank count and does not depend on policy concentration), and we report the ratio between the learned worst-case constraint's damage and matched random damage.

| Game     | Reachable P0 states | Damage ratio |
|----------|--------------------:|-------------:|
| Leduc    |                 144 |       `2.2×` |
| Leduc-5  |                 390 |       `4.6×` |
| Leduc-10 |               1,530 |       `4.7×` |
| Leduc-20 |               5,900 |       `4.8×` |
| Leduc-30 |              12,248 |       **`8.4×`** |
| Leduc-50 |              26,293 |       **`7.6×`** |

Log-linear regression (excluding Kuhn's 2-action ceiling) gives `R² = 0.70` (`r = 0.84`, slope ≈ `2.0` damage-ratio units per decade of state count). Adversarial variance stays tight at the largest scales (`±0.15` at Leduc-50). Reproduced by `experiments/leduc{5,10,20,30,50}_scale.py` and `experiments/state_counts_uniform.py`; regression in `experiments/scaling_regression.py`.

## 4. Defenses (what worked, what didn't)

| Defense                                 | Leduc attacked reward | Δ vs standard |
|-----------------------------------------|----------------------:|--------------:|
| Standard self-play (baseline)           |              `−2.45`  | —             |
| Stochastic action dropout (`p=0.2`)     |              `−1.82`  | `+0.63`       |
| Static random-mask ensemble             |              `−2.64`  | `−0.19`       |
| Adversarial co-training (no `CAC_v` reg) |             `−2.36`  | `−0.30`       |
| `CAC_v`-guided redundancy (tuned `λ`)   |              `−1.43`  | **`+1.02`**   |
| Mask-aware C-MDP (observes availability)|              `−0.18`  | **`+2.27`**   |

Notable takeaways:

- **Adversarial co-training is a negative result.** Just exposing the victim to a learned constraint during training without changing the policy structure (no value-gap regularisation) does not help and adds variance. The victim over-fits to specific constraint patterns seen in training and loses to a freshly-trained constraint at evaluation.
- **`CAC_v`-guided redundancy** Pareto-improves over standard self-play. Sweeping the regularisation strength `λ` traces a clean robustness–exploitation frontier (Figure in `paper/eumas/figures/robustness_frontier.pdf`). At `λ ≈ 0.01` Leduc adversarial reward goes from `−2.26` to `−1.43` (36% damage reduction) with minimal cost to clean reward.
- **Mask-aware C-MDP** is a much stronger defense. Augmenting the agent's state with a binary availability vector and training against structured random availability patterns produces an agent that is essentially robust to single-action constraints (92% damage reduction in Leduc). This requires a stronger threat model — the agent must observe its own availability vector — but exactly that assumption is realistic in most non-malicious settings (faults, rate limits, sandboxing).

The dichotomy that emerges is clean: availability constraints are devastating against *unaware* self-play agents but largely defensible once awareness is restored. From a MAS-design perspective the prescription is to treat action availability as a first-class observation channel rather than something hidden inside the environment.

## 5. Causal intervention

To distinguish correlation from causation for `CAC_v`, we train two agents on the same game with the same training budget, differing only in whether a `CAC_v` regularizer is active (compressed vs redundant agents). Then we attack both with the same learned constraint policy. In Kuhn the redundant agent has smaller mean value gaps (`0.625` vs `0.840`) and demonstrably better attacked reward (`−0.619` vs `−0.763`). In Leduc the effect is smaller and visible only when the regularisation strength is tuned (the robustness frontier sweep above). Reproducible by `experiments/cacv_intervention.py`.

## 6. Exploitability evidence (best-response oracle)

To make sure the head-to-head reward numbers are not just a story about opponent co-adaptation, we compute exact best-response value for the trained P0 policy by enumerating the Kuhn game tree. Pre-attack exploitability is `0.43 ± 0.14` per hand; post-attack it is `1.20 ± 0.06` per hand — a `2.8×` increase. This matches the head-to-head reward gap and rules out the "the opponent just got better" explanation: the agent's own policy quality degrades. Reproducible by `experiments/kuhn_exploitability.py`.

## 7. Cooperative MAS (Hanabi)

OpenSpiel's Python Hanabi bindings are not available on Windows without a C++ build, so we implemented a self-contained Hanabi-V2 variant (3 colours × 5 ranks, hand size 3, standard `[3, 2, 2, 2, 1]` rank distribution per colour, 14 actions). Tabular self-play saturates around `+1.3 / +15` team reward in 8k episodes — strong enough that structural removals affect coordination metrics but not strong enough for a learned worst-case constraint to find brittle conventions. Two clean signals do emerge:

- Removing all hint actions (`comm_removal`) cuts P0 action entropy from `2.16` to `1.57` bits.
- Removing only rank-hints (keeping colour) cuts hint entropy from `2.93` to `2.62` bits, forcing a colour-only convention.

This is reported in the paper as preliminary cooperative-MAS evidence; the obvious follow-up is to run the full Hanabi-learning-environment with neural function approximation. Reproducible by `experiments/hanabi_v2_masking.py`.

## 8. Theoretical scope

The paper carries five propositions, all proved in short form in the main text or appendix:

1. **DEA Convergence**. Under `CAC_w = 0` in self-play Q-learning, the victim collapses to forced actions, the opponent's MDP becomes stationary and converges, and the resulting `(σ*, BR(σ*))` is a stable fixed point.
2. **Damage bound**. The reach-weighted Q-gap decomposition of value loss; identifies the greedy `ρ·δ` heuristic.
3. **Fragility and redundancy**. Lower and upper bounds: high-`δ` mass implies unavoidable exploitability; bounded `δ` implies bounded damage. Grounds the defensive intuition behind `CAC_v` regularisation.
4. **`CAC_v` lower bound on exploitability**. Worst-case damage `≥ CAC_v^(k)(σ)` independent of how the constraint policy is parameterised. This is what makes `CAC_v` a true property of the agent's policy rather than an artifact of REINFORCE.
5. **Greedy near-optimality under weak coupling** (appendix). Standard `(1 − 1/e)` submodular guarantee under bounded cross-state interactions, plus a one-step downstream extension of the damage bound.

We also prove NP-hardness of optimal mask selection via reduction from Weighted Maximum Coverage (appendix). This justifies using learned constraint policies rather than exact combinatorial search.

The theory is honest: it explains why the phenomenon happens and gives a defensive design target, but it does not claim deep equilibrium characterisation.

## 9. What was tried and didn't work

Negative results worth documenting:

- **OpenSpiel real Hanabi on Windows** — the PyPI `open_spiel` package only ships C++ source on Windows; building requires Visual Studio + CMake. Time-budget too high. Used a custom Hanabi-V2 variant instead.
- **Adversarial co-training without value-gap regularisation** — does not improve robustness; the victim over-fits to specific constraint patterns. Reported as a negative result.
- **Pure scale push to "real poker" abstractions** — would need GPU weeks of compute. Out of scope; explicitly listed as future work.

## 10. Submission history

- **NeurIPS 2026**: desk reject. Reviewer feedback noted the framing was too "ML robustness" for the venue and the defense story was incomplete at scale.
- **EUMAS pivot**: reframed to lead with structural robustness in multi-agent systems, contingent decision capacity as a design target, action availability as a coordination/control primitive. Section titles, abstract, and intro rewritten; `CAC_v` and the mask-aware C-MDP defense promoted. Switched LaTeX format to Springer LNCS via the official `llncs.cls` (v2.26) and `splncs04.bst`. Hanabi-V2 result moved into the main body (not just appendix) to surface the cooperative-MAS angle.

## 11. Repository state

After cleanup, the GitHub repository is code-only. Paper sources (`paper/`), the anonymised supplementary mirror (`supplementary_material/`), and the local experiment outputs (`results/`) are all `.gitignore`d. The paper itself lives separately (Overleaf / local). Compiled PDFs, LaTeX intermediates, and `.zip` archives are also ignored.

If you cloned this repository fresh and want to reproduce a result, run any of the scripts named in [README.md](README.md). Every script is self-contained and sets its own seeds.
