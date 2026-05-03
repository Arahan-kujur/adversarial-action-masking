"""Controls for when the victim sees the mask.

Compares:
1. No mask.
2. Evaluation-only masking: victim trains normally, mask applied only at test.
3. Continued masked training: victim trains normally, then adapts under a fixed mask.
4. Mask-aware from scratch: victim trains from scratch under the fixed learned mask.

The mask is learned once per seed on a separate victim, then frozen for the
three evaluation protocols.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adversary.mask_utils import evaluate_agent
from adversary.masking_policy import AdversarialMask
from core.agents.q_learning import QLearningAgent
from core.envs.leduc_poker import LeducPokerEnv, MaskedLeducPoker, NUM_ACTIONS
from core.training.selfplay import play_episode


SEEDS = list(range(5))
PRETRAIN = 12000
ADAPT = 12000
EVAL = 3000


def train_full(seed: int):
    rng = np.random.default_rng(seed)
    env = MaskedLeducPoker(LeducPokerEnv())
    agent = QLearningAgent(num_actions=NUM_ACTIONS, alpha=0.1, epsilon=0.15)
    for _ in range(PRETRAIN):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)
    return env, agent, rng


def learn_mask(seed: int):
    env, agent, rng = train_full(seed + 1000)
    adv = AdversarialMask(target_player=0, num_actions=NUM_ACTIONS, lr=0.01)
    env.set_mask(adv.mask_fn)
    for _ in range(15):
        batch = []
        for _ in range(400):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)
            batch.append((t, r))
        adv.update(batch)
    return adv


def train_under_mask(seed: int, mask_fn, episodes: int):
    rng = np.random.default_rng(seed)
    env = MaskedLeducPoker(LeducPokerEnv())
    agent = QLearningAgent(num_actions=NUM_ACTIONS, alpha=0.1, epsilon=0.15)
    env.set_mask(mask_fn)
    for _ in range(episodes):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)
    return env, agent, rng


def continue_under_mask(env, agent, rng, mask_fn):
    env.set_mask(mask_fn)
    for _ in range(ADAPT):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)


def main():
    print("Mask timing controls (Leduc QL)", flush=True)
    print("=" * 60, flush=True)
    results = {
        "none": [],
        "eval_only": [],
        "continued_masked": [],
        "mask_aware_scratch": [],
    }

    for seed in SEEDS:
        adv = learn_mask(seed)
        env, agent, rng = train_full(seed)
        results["none"].append(evaluate_agent(env, agent, rng, EVAL))
        results["eval_only"].append(evaluate_agent(env, copy.deepcopy(agent), rng, EVAL, mask_fn=adv.mask_fn))

        cont_agent = copy.deepcopy(agent)
        cont_env = MaskedLeducPoker(LeducPokerEnv())
        continue_under_mask(cont_env, cont_agent, rng, adv.mask_fn)
        results["continued_masked"].append(evaluate_agent(cont_env, cont_agent, rng, EVAL, mask_fn=adv.mask_fn))

        aware_env, aware_agent, aware_rng = train_under_mask(seed + 2000, adv.mask_fn, PRETRAIN + ADAPT)
        results["mask_aware_scratch"].append(evaluate_agent(aware_env, aware_agent, aware_rng, EVAL, mask_fn=adv.mask_fn))
        print(f"  seed={seed} done", flush=True)

    for mode, vals in results.items():
        vals = np.asarray(vals, dtype=float)
        mean = vals.mean()
        ci = 1.96 * vals.std() / np.sqrt(len(vals))
        print(f"  {mode:>20s}: {mean:+.3f} +/- {ci:.3f}", flush=True)


if __name__ == "__main__":
    main()
