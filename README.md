# Adversarial Action Masking in Self-Play RL

An adversary learns to remove actions from an RL agent mid-training,
exploiting co-adaptation dynamics in self-play. We study which masking
strategies cause the most damage and whether agents can adapt.

## Quick Start

```bash
pip install -r requirements.txt

# Train agent, then evaluate under fixed mask
python experiments/train_selfplay.py

# Train adversarial masking policy
python experiments/train_adversary.py

# Compare all masking strategies
python experiments/evaluate_attack.py
```

## What is Adversarial Action Masking?

Standard action masking removes invalid actions (e.g., illegal moves).
**Adversarial** action masking asks: what if an attacker chooses *which*
actions to remove, in order to maximise the agent's loss?

This is relevant when:
- Hardware failures disable specific actuators
- An adversary controls which API endpoints are available
- Regulatory changes remove actions strategically

We implement a bi-level optimisation:
- **Inner loop**: RL agent trains via self-play under the mask
- **Outer loop**: adversary learns which actions to remove to minimise agent reward

## Masking Strategies

| Strategy | Description |
|---|---|
| No mask | Baseline -- full action space |
| Random | Remove actions with probability p per step |
| Fixed | Always remove a specific action (e.g., BET) |
| Adversarial | Learned policy that chooses which action to remove per state |

## Project Structure

```
core/
  envs/kuhn_poker.py       Kuhn Poker env + MaskedKuhnPoker wrapper
  agents/q_learning.py      Tabular Q-learning agent
  training/selfplay.py      Self-play training loop
adversary/
  masking_policy.py          Masking strategies (random, fixed, adversarial)
  mask_utils.py              Evaluation and analysis utilities
experiments/
  train_selfplay.py          Train + evaluate under fixed mask
  train_adversary.py         Bi-level adversarial training
  evaluate_attack.py         Compare all strategies side-by-side
configs/                     Experiment configurations
results/                     Output (gitignored CSVs)
```

## Requirements

Python 3.10+. Only dependency: `numpy`.
