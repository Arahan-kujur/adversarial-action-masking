"""Cross-algorithm and cross-game transfer experiments.

1. Train adversary on QL victim in Kuhn, test on PPO/NFSP/DQN victims in Kuhn
2. Train adversary on Kuhn QL, test on Leduc QL (cross-game)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.envs.kuhn_poker import KuhnPokerEnv, MaskedKuhnPoker
from core.envs.leduc_poker import LeducPokerEnv, MaskedLeducPoker, NUM_ACTIONS as LEDUC_ACTIONS
from core.agents.q_learning import QLearningAgent
from core.agents.nfsp import NFSPAgent
from core.training.selfplay import play_episode
from adversary.masking_policy import AdversarialMask
from adversary.mask_utils import evaluate_agent

SEEDS = list(range(5))


def train_adversary_kuhn(seed):
    """Train tabular adversary on QL victim in Kuhn. Return the mask_fn."""
    rng = np.random.default_rng(seed)
    env = MaskedKuhnPoker(KuhnPokerEnv())
    agent = QLearningAgent(num_actions=2, alpha=0.1, epsilon=0.15)
    for _ in range(10000):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)
    adv = AdversarialMask(target_player=0, num_actions=2, lr=0.01)
    env.set_mask(adv.mask_fn)
    for _ in range(20):
        batch = []
        for _ in range(500):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)
            batch.append((t, r))
        adv.update(batch)
    return adv


def test_on_victim(agent_class, agent_kwargs, mask_fn, seed):
    """Train fresh victim, then test under transferred mask."""
    rng = np.random.default_rng(seed)
    env = MaskedKuhnPoker(KuhnPokerEnv())
    agent = agent_class(**agent_kwargs)
    for _ in range(10000):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)
    pre = evaluate_agent(env, agent, rng, 3000)
    env.set_mask(mask_fn)
    for _ in range(10000):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)
    post = evaluate_agent(env, agent, rng, 3000, mask_fn=mask_fn)
    return pre, post


def main():
    print("=" * 65, flush=True)
    print("Cross-Algorithm Transfer (adversary trained on Kuhn QL)", flush=True)
    print("=" * 65, flush=True)

    adv = train_adversary_kuhn(42)

    victims = {
        "QL": (QLearningAgent, {"num_actions": 2, "alpha": 0.1, "epsilon": 0.15}),
        "NFSP": (NFSPAgent, {"num_actions": 2, "alpha": 0.1, "epsilon": 0.15, "eta": 0.1}),
    }

    for name, (cls, kwargs) in victims.items():
        pre_vals, post_vals = [], []
        for seed in SEEDS:
            pre, post = test_on_victim(cls, kwargs, adv.mask_fn, seed)
            pre_vals.append(pre)
            post_vals.append(post)
        mp = np.mean(pre_vals)
        mpo = np.mean(post_vals)
        ci = 1.96 * np.std(post_vals) / np.sqrt(len(post_vals))
        print(f"  {name:>6s}: pre={mp:+.3f}  post={mpo:+.3f} +/-{ci:.3f}  "
              f"delta={mpo-mp:+.3f}", flush=True)

    # Cross-game: Kuhn adversary -> Leduc
    print("\n" + "=" * 65, flush=True)
    print("Cross-Game Transfer (Kuhn adversary -> Leduc QL)", flush=True)
    print("=" * 65, flush=True)

    pre_vals, post_vals = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        env = MaskedLeducPoker(LeducPokerEnv())
        agent = QLearningAgent(num_actions=LEDUC_ACTIONS, alpha=0.1, epsilon=0.15)
        for _ in range(15000):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)
        pre = evaluate_agent(env, agent, rng, 3000)

        env.set_mask(adv.mask_fn)
        for _ in range(15000):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)
        post = evaluate_agent(env, agent, rng, 3000, mask_fn=adv.mask_fn)
        pre_vals.append(pre)
        post_vals.append(post)

    mp = np.mean(pre_vals)
    mpo = np.mean(post_vals)
    ci = 1.96 * np.std(post_vals) / np.sqrt(len(post_vals))
    print(f"  Leduc QL: pre={mp:+.3f}  post={mpo:+.3f} +/-{ci:.3f}  "
          f"delta={mpo-mp:+.3f}", flush=True)
    print(f"\n  (Kuhn adversary uses 2-action state keys; Leduc has different",
          flush=True)
    print(f"   state encoding -> transfer only works if state keys overlap)",
          flush=True)


if __name__ == "__main__":
    main()
