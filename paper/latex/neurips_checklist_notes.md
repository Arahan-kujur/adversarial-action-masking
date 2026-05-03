# NeurIPS Paper Checklist Answers

These notes are written to match the official NeurIPS Paper Checklist:
<https://neurips.cc/public/guides/PaperChecklist>. They can be copied into the
checklist section or submission form. Keep anonymity requirements in mind if
submitting through a double-blind review system.

## 1. Claims

**Answer: Yes.**

The abstract and introduction state that the contribution is adversarial action
removal in discrete self-play RL. Claims are limited to the tested setting:
discrete action spaces, poker variants up to Leduc-20, gridworld/resource
collection domains, and the listed victim algorithms. The paper explicitly
frames scaling as an empirical trend, not a scaling law.

## 2. Limitations

**Answer: Yes.**

The paper includes a Limitations section. It states that the method is scoped to
discrete action spaces, that continuous-action analogues require region
exclusion, and that the theory provides local/sufficient bounds rather than a
complete characterization of optimal adversaries.

## 3. Theory, Assumptions and Proofs

**Answer: Yes.**

The propositions state their assumptions. Proposition 1 covers the CACw=0
degenerate endpoint. Proposition 2 gives a local reach-weighted value-gap bound.
The paper explicitly states that the greedy/submodularity discussion depends on
an independence approximation and should be interpreted as explanatory, not as a
general approximation guarantee.

## 4. Experimental Result Reproducibility

**Answer: Yes.**

All experiments are implemented as standalone scripts under `experiments/`, with
fixed seeds and documented hyperparameters in the appendix. The paper includes a
Reproducibility Statement and points to the relevant scripts.

## 5. Open Access to Data and Code

**Answer: Yes.**

The repository contains all environment, agent, adversary, and experiment code.
There are no external datasets. The main reproducibility scripts include:

- `experiments/leduc20_scale.py`
- `experiments/reviewer_strengthening.py`
- `experiments/matched_l0_control.py`
- `experiments/mask_timing_controls.py`
- `experiments/mask_ensemble_defense.py`
- `experiments/generate_neurips_figures.py`

For anonymous submission, use an anonymized repository or supplemental archive.

## 6. Experimental Setting / Details

**Answer: Yes.**

The main paper describes the self-play protocol, masking protocol, adversary
training loop, and masking granularity. Appendix C lists hyperparameters for
Q-learning, PPO, DQN, neural NFSP, and adversaries. Environment definitions are
implemented in `core/envs/`.

## 7. Experiment Statistical Significance

**Answer: Yes.**

Main tables report 95% confidence intervals across seeds where applicable.
The text states that reported uncertainty is over seeded runs. Some auxiliary
tables report deterministic summaries or diagnostic means; those are described
as diagnostics rather than primary claims.

## 8. Experiments Compute Resource

**Answer: Yes.**

The appendix includes a representative compute/sample-cost table. Experiments
run on CPU with PyTorch used for DQN, neural NFSP, and neural adversary MLPs.
No large GPU cluster is required. The largest reported run is Leduc-20 with five
seeds, 30k victim pre-training episodes, and 25 adversary outer iterations with
500 inner episodes each.

## 9. Code of Ethics

**Answer: Yes.**

The work is a security/robustness study of RL systems. The paper focuses on
identifying a vulnerability and motivating defenses. No human subjects, private
data, or deployed systems are involved.

## 10. Broader Impacts

**Answer: Yes.**

The paper discusses practical implications: actuator failures, API removals,
and capability constraints. It also highlights possible misuse because targeted
capability removal can degrade multi-agent systems. The defensive direction is
to preserve redundancy at high-reach/high-CACv states and monitor action
availability changes.

## 11. Safeguards

**Answer: N/A.**

No pretrained models or high-risk deployed models are released. The repository
contains small research environments and experiment scripts.

## 12. Licenses

**Answer: Partially / to be finalized before public release.**

The code is original except for standard Python dependencies (`numpy`, `torch`,
`matplotlib`) and the NeurIPS style file. No external datasets are used. The
repository currently does not include a license file; a license should be added
before public release.

## 13. Assets

**Answer: Yes.**

The released assets are code, custom environments, and paper sources. The README
documents the repository layout, dependencies, run commands, and scope. No data
assets or trained checkpoints are required.

## 14. Crowdsourcing and Research with Human Subjects

**Answer: N/A.**

No crowdsourcing or human-subject experiments are used.

## 15. IRB Approvals

**Answer: N/A.**

No human-subject research is conducted.

## 16. Declaration of LLM Usage

**Answer: N/A for core methodology.**

The core methods, experiments, and analysis do not use LLMs. If LLM-assisted
editing or formatting was used, it is not part of the scientific method and does
not affect the reported results.
