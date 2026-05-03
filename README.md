# Adversarial Action Removal in Self-Play RL

This repository contains the experiments and paper source for **Adversarial Action Removal in Self-Play Reinforcement Learning**.

The paper studies a structural adversarial attack: instead of perturbing observations or replacing chosen actions, an attacker removes legal actions from the victim's available action set before action selection. The main finding is that targeted action removal is substantially more damaging than random masking or learned action perturbation, and the effect persists across algorithms, domains, and game sizes.

## Highlights

- New attack surface: adversarial removal of legal actions.
- Bi-level adversary: victim trains under a mask; adversary learns which actions to remove.
- Mechanism: reach-weighted and value-weighted contingent action capacity (`CAC_w`, `CAC_v`).
- Scale: poker variants from Kuhn (6 victim information states) to Leduc-20 (5,531 victim information states).
- Algorithms: Q-learning, PPO, NFSP, neural NFSP, and DQN.
- Domains: poker, competitive gridworld, and resource collection.
- NeurIPS-style paper source and reproducibility notes included.

## Key Results

### Scaling With Game Size

| Game | Victim Info States | Victim | Adversarial / Random Damage |
|---|---:|---|---:|
| Leduc | ~50 | DQN | 2.2x |
| Leduc-5 | 389 | DQN | 4.6x |
| Leduc-10 | 1,496 | DQN | 4.7x |
| Leduc-20 | 5,531 | DQN | 4.8x |

The adversary remains effective as game complexity grows. The largest run, Leduc-20, reaches `-3.00 +/- 0.15` victim reward versus `-0.63 +/- 0.08` for random masking.

### Reviewer-Response Controls

| Control | Result |
|---|---|
| Public-information adversary | Still beats random in Leduc: `-1.71` vs `-0.98` |
| Matched-L0 random baseline | Same support size, adversary still 2.24x worse |
| CACv-greedy oracle | Stronger than random and short-trained learned adversary |
| Separate-network DQN | Collapse persists without shared parameters |
| Evaluation-only masking | Immediate damage: `-0.58` after normal training |
| Mask-aware training from scratch | Still collapses: `-2.71` |
| Action-dropout defense | Helps modestly: `-1.82` vs standard `-2.45` |
| Random mask-ensemble defense | Does not help: `-2.64` |

## Repository Layout

```text
adversary/
  masking_policy.py        Random, fixed, and learned action-removal policies
  mask_utils.py            Evaluation and mask statistics

core/
  agents/
    q_learning.py          Tabular Q-learning
    ppo.py                 Tabular PPO
    nfsp.py                Tabular NFSP
    neural_nfsp.py         Neural NFSP
    dqn.py                 DQN + encoders for Kuhn/Leduc/Leduc-N
  envs/
    kuhn_poker.py          Kuhn Poker
    leduc_poker.py         Leduc Poker
    leduc_n.py             Leduc-N scale variants
    gridworld.py           Competitive gridworld
    resource_collection.py Resource collection game
  training/
    selfplay.py            Shared self-play loop

experiments/
  leduc20_scale.py         Largest DQN scaling run
  leduc10_scale.py         Leduc-10 scaling run
  leduc5_scale.py          Leduc-5 scaling run
  neural_nfsp_leduc5.py    Neural NFSP under attack
  reviewer_strengthening.py Public-info, CACv oracle, L0, separate DQN, dropout
  matched_l0_control.py    Strict matched-L0 random control
  mask_timing_controls.py  Evaluation-only and mask-aware victim controls
  mask_ensemble_defense.py Mask-ensemble defense baseline
  generate_neurips_figures.py Figure generation script

paper/
  paper.md                 Markdown paper summary
  latex/
    main.tex               NeurIPS-style LaTeX source
    main.pdf               Compiled PDF
    figures/               Generated figures
    references.bib         Bibliography
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows PowerShell
pip install -r requirements.txt
```

Dependencies:

- `numpy`
- `torch`
- `matplotlib`

## Quick Start

Run the smallest attack comparison:

```bash
python experiments/evaluate_attack.py
```

Run the main scaling experiments:

```bash
python experiments/leduc5_scale.py
python experiments/leduc10_scale.py
python experiments/leduc20_scale.py
```

Run the reviewer-response controls:

```bash
python experiments/reviewer_strengthening.py
python experiments/matched_l0_control.py
python experiments/mask_timing_controls.py
python experiments/mask_ensemble_defense.py
```

Regenerate figures:

```bash
python experiments/generate_neurips_figures.py
```

Compile the paper:

```bash
cd paper/latex
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Paper

The NeurIPS-style source is in [`paper/latex/main.tex`](paper/latex/main.tex), with the compiled PDF at [`paper/latex/main.pdf`](paper/latex/main.pdf).

The main body is kept within the NeurIPS 9-page target; supporting ablations, hyperparameters, normalization bounds, and learning curves are in the appendix.

## Reproducibility

All reported experiments are standalone Python scripts under `experiments/`. Seeds are fixed inside the scripts. The largest experiment (`experiments/leduc20_scale.py`) uses five seeds, 30k victim pre-training episodes, and 25 adversary outer iterations with 500 victim-training episodes per adversary update.

## Citation

If you use this code, cite the repository or paper draft:

```bibtex
@misc{kujur2026adversarialactionremoval,
  title={Adversarial Action Removal in Self-Play Reinforcement Learning},
  author={Kujur, Arahan},
  year={2026},
  note={Preprint}
}
```

## License

No license file is currently included. Add a license before public release if you plan to distribute or accept contributions.
