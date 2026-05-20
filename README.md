# Contingent Decision Capacity

Code for the paper **Contingent Decision Capacity: Structural Robustness of Self-Play Agents to Action Availability Constraints** (submitted to EUMAS). An earlier version was titled *When Actions Disappear: Adversarial Action Removal in Self-Play Reinforcement Learning*.

The paper studies a structural perturbation of multi-agent systems: at decision time, an agent's available actions are reduced (by faults, sandboxing, rate-limiting, regulatory restrictions, or strategic constraints). It introduces **contingent action capacity** (`CAC_w` and its value-weighted refinement `CAC_v`) as a way to measure how much strategic flexibility an agent retains under such constraints, and shows that self-play strategies concentrate value at a small number of pivotal information states, which makes them disproportionately fragile to targeted availability constraints.

## Headline results

- **Phenomenon**: across six Leduc-N variants (144 to 26,293 reachable P0 information states), a learned worst-case availability constraint causes 2.2× to **8.4×** more damage than random unavailability of equal support, with log-linear scaling regression `R² = 0.70`.
- **Mechanism**: `CAC_v` lower-bounds worst-case damage independent of the constraint policy's parametric class (Prop. 4); empirically correlates with reward at `r = 0.81`.
- **Defenses**:
  - Uniform random dropout helps modestly (Leduc `-1.82` vs `-2.45` undefended).
  - Static random-mask ensembles fail.
  - `CAC_v`-guided redundancy training Pareto-improves robustness (Leduc `-2.26 → -1.43`, **36% damage reduction**).
  - A constrained-MDP-style **mask-aware** agent (observes its own availability vector) reaches `-0.18` (**92% damage reduction**) — clean dichotomy: the attack devastates unaware agents but is largely defensible once awareness is restored.
- **Cooperative MAS**: in a Hanabi-V2 variant (3 colours × 5 ranks, hand size 3), structural removal of communication actions cleanly degrades convention metrics (action entropy 2.16 → 1.57 bits).
- **Cross-domain**: same phenomenon holds in a competitive gridworld and a resource-collection game.
- **Best-response oracle**: in Kuhn, exploitability against a true best-responder rises from `0.43` to `1.20` per hand after the attack (`2.8×` increase) — consistent with head-to-head reward, confirming intrinsic policy degradation rather than opponent co-adaptation.

A full chronological narrative of the project (NeurIPS submission, desk reject, EUMAS pivot, ablations, defense work) is in [Report.md](Report.md).

## Repository layout

```text
adversary/
  masking_policy.py        Random, fixed, and learned action-availability policies
  mask_utils.py            Evaluation helpers and mask diagnostics

core/
  agents/
    q_learning.py          Tabular Q-learning (with optional CACv-regularization)
    ppo.py                 Tabular PPO
    nfsp.py                Tabular NFSP
    neural_nfsp.py         Neural NFSP
    dqn.py                 DQN + encoders for Kuhn / Leduc / Leduc-N (any N)
  envs/
    kuhn_poker.py          Kuhn Poker
    leduc_poker.py         Leduc Poker
    leduc_n.py             Leduc-N scale variants
    gridworld.py           Competitive 5x5 gridworld (prey/predator)
    resource_collection.py 4x4 resource competition
    hanabi_small.py        Hanabi-Small + Hanabi-V2 cooperative variants
  training/
    selfplay.py            Shared self-play loop

experiments/
  Scaling
    leduc5_scale.py, leduc10_scale.py, leduc20_scale.py
    leduc30_scale.py, leduc50_scale.py
    state_counts_uniform.py    Reachable state counts under uniform random play
    scaling_regression.py      Log-linear regression of damage ratio vs game size

  Defenses
    mask_ensemble_defense.py
    cacv_regularized_defense.py
    cacv_intervention.py              Compressed vs redundant victim
    robustness_frontier.py            Sweep regularization strength
    plot_robustness_frontier.py
    adversarial_cotraining_defense.py
    mask_aware_robust_defense.py      Constrained-MDP-style baseline

  Mechanism
    cacv_metric.py, cacw_correlation.py
    targeting_analysis.py, attack_efficiency.py
    rarl_comparison.py, learned_perturbation.py
    matched_l0_control.py, budget_fairness.py, budget_sweep_detailed.py
    reviewer_strengthening.py         Public-info, CACv oracle, separate-DQN, dropout
    strategic_concentration_profiles.py
    aggregate_strategic_concentration.py
    final_strategic_compression_analysis.py
    summarize_crt_frontier.py

  Cooperative MAS
    hanabi_small_masking.py
    hanabi_v2_masking.py

  Reviewer-response controls
    kuhn_exploitability.py            Best-response oracle exploitability

  Other
    cross_comparison.py, cross_domain.py, cross_domain2.py
    cross_transfer.py, attack_generalization.py
    dqn_leduc_full.py, neural_adversary.py
    nfsp_victim.py, nfsp_leduc.py, nfsp_leduc5.py, neural_nfsp_leduc5.py
    separate_networks.py, vulnerability_comparison.py
    learning_curves.py, gap_vs_size.py, ablation_heuristic.py
    minimal_attack.py, evaluate_attack.py
    train_selfplay.py, train_adversary.py
    generate_neurips_figures.py, generate_table.py
```

The compiled paper, supplementary submission mirror, and generated result CSVs/figures are all excluded by `.gitignore`; this repository is code-only.

## Installation

Python 3.10+ recommended.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
# or
source .venv/bin/activate       # Linux/macOS

pip install -r requirements.txt
```

Runtime dependencies are `numpy`, `torch`, and `matplotlib`.

## Quick start

Smallest end-to-end example (Kuhn poker, learned constraint policy):

```bash
python experiments/minimal_attack.py
```

Headline scaling table:

```bash
python experiments/leduc5_scale.py
python experiments/leduc10_scale.py
python experiments/leduc20_scale.py
python experiments/leduc30_scale.py
python experiments/leduc50_scale.py
python experiments/state_counts_uniform.py
python experiments/scaling_regression.py
```

Defenses (CACv-guided redundancy + mask-aware C-MDP):

```bash
python experiments/cacv_regularized_defense.py
python experiments/robustness_frontier.py
python experiments/plot_robustness_frontier.py
python experiments/mask_aware_robust_defense.py
```

Best-response oracle exploitability (Kuhn):

```bash
python experiments/kuhn_exploitability.py
```

Cooperative benchmark:

```bash
python experiments/hanabi_v2_masking.py
```

All scripts set their own random seeds internally and write any artifacts under `results/` (ignored by git).

## Reproducibility

Every reported result comes from a standalone script under `experiments/`. The largest run (`leduc50_scale.py`) uses five seeds, 30k pre-training episodes, and 25 outer × 500 inner constraint-training episodes; on a desktop CPU it finishes in ~8 minutes per seed (~40 minutes total).

For the cooperative Hanabi-V2 result, three seeds and ~8k tabular self-play episodes per seed are sufficient for the structural removal effect to show up in convention metrics; the learned worst-case constraint requires stronger victims (neural function approximation) and is noted as future work in the paper.

## Citation

```bibtex
@misc{kujur2026contingentcapacity,
  title  = {Contingent Decision Capacity: Structural Robustness of Self-Play Agents to Action Availability Constraints},
  author = {Kujur, Arahan},
  year   = {2026},
  note   = {Preprint; submitted to EUMAS.}
}
```

## License

No license file is included yet. Add one (MIT, Apache-2.0, etc.) before public distribution if you want to clarify reuse terms.
