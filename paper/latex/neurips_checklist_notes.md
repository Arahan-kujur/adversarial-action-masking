# NeurIPS Checklist Notes

These notes are not part of the paper body. They collect the answers needed for
the official NeurIPS checklist.

## Claims

- Main claims are empirical and bounded to discrete action-space games.
- Scaling is described as an empirical trend, not a scaling law.
- The theory is explicitly scoped as sufficient/local, not a complete
  characterization of the optimal adversary.

## Limitations

- The paper includes a Limitations section.
- It explicitly states that continuous-action domains require a different
  region-exclusion formalism and are not claimed by this work.

## Theory

- Assumptions are stated in the propositions and in the theoretical scope
  paragraph.
- The greedy approximation relies on an independence/submodularity condition
  and this is disclosed.

## Experiments

- All training scripts are under `experiments/`.
- Seeds are fixed in the scripts.
- Main hyperparameters are listed in Appendix C.
- Reported uncertainty is 95% confidence intervals unless otherwise noted.
- Largest experiment: `experiments/leduc20_scale.py`.
- Reviewer-response ablations are in `experiments/reviewer_strengthening.py`:
  public-information adversary, CACv-greedy oracle, exact L0 diagnostics,
  separate-network DQN, and action-dropout defense.

## Compute

- Experiments run on CPU with PyTorch for DQN/neural NFSP/adversary MLPs.
- No large GPU cluster is required.

## Data

- No external datasets are used.
- Environments are implemented from scratch in `core/envs/`.

## Code

- Repository contains all code needed to reproduce experiments.
- Figure generation is in `experiments/generate_neurips_figures.py`.
