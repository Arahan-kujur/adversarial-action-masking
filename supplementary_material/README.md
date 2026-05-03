# Supplementary Material

This folder contains anonymized supplementary code for the paper
**"When Actions Disappear: Adversarial Action Removal in Self-Play Reinforcement Learning."**

It is intended to be zipped and uploaded as NeurIPS supplementary material. Written appendices are not included here; they are part of the main paper PDF.

## Contents

```text
adversary/
  masking_policy.py        Random, fixed, and learned action-removal masks
  mask_utils.py            Evaluation and mask diagnostics

core/
  agents/                  Q-learning, PPO, NFSP, neural NFSP, DQN
  envs/                    Kuhn, Leduc, Leduc-N, gridworld, resource collection
  training/                Shared self-play loop

experiments/
  leduc20_scale.py         Largest scaling experiment
  leduc10_scale.py         Leduc-10 scale experiment
  leduc5_scale.py          Leduc-5 scale experiment
  reviewer_strengthening.py Public-info, CACv oracle, L0, separate-DQN, dropout controls
  matched_l0_control.py    Strict matched-L0 random control
  mask_timing_controls.py  Evaluation-only and mask-aware training controls
  mask_ensemble_defense.py Mask-ensemble defense baseline
  generate_neurips_figures.py Figure generation from completed results

requirements.txt           Python dependencies
```

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies:

- `numpy`
- `torch`
- `matplotlib`

## Reproducing Main Results

Run the main scale experiments:

```bash
python experiments/leduc5_scale.py
python experiments/leduc10_scale.py
python experiments/leduc20_scale.py
```

Run the main reviewer-response controls:

```bash
python experiments/reviewer_strengthening.py
python experiments/matched_l0_control.py
python experiments/mask_timing_controls.py
python experiments/mask_ensemble_defense.py
```

Run cross-domain experiments:

```bash
python experiments/cross_domain.py
python experiments/cross_domain2.py
```

Regenerate paper figures:

```bash
python experiments/generate_neurips_figures.py
```

## Notes

- All environments are synthetic and implemented from scratch.
- No external datasets or pretrained models are required.
- Scripts set their own random seeds internally.
- The largest reported run is `experiments/leduc20_scale.py`.
- This supplementary folder intentionally excludes paper PDFs, LaTeX sources, and written appendices.

