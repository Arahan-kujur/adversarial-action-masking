"""Mask-ensemble robust training baseline.

Victim trains against an ensemble of random state/action masks rather than
independent action dropout. This is a stronger defense because the agent sees
persistent state-conditioned capability losses during training.
"""
from __future__ import annotations

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
DEFENSE_EPISODES = 12000
EVAL = 3000


def collect_states(agent, rng, episodes=3000):
    env = MaskedLeducPoker(LeducPokerEnv())
    states = set()
    for _ in range(episodes):
        env.reset(rng=rng)
        while not env.is_terminal:
            p = env.current_player
            info = env.info_state(p)
            if p == 0:
                states.add(info)
            a = agent.select_action(info, env.legal_actions(), rng)
            env.step(a)
    return sorted(states)


def make_persistent_random_mask(states, seed, budget_fraction=0.6):
    rng = np.random.default_rng(seed)
    k = max(1, int(len(states) * budget_fraction))
    chosen = set(rng.choice(states, size=min(k, len(states)), replace=False))
    removed = {s: int(rng.integers(0, NUM_ACTIONS)) for s in chosen}

    def mask_fn(info_state, legal_actions, player):
        if player != 0 or info_state not in removed or len(legal_actions) <= 1:
            return legal_actions
        a = removed[info_state]
        if a not in legal_actions:
            a = legal_actions[-1]
        filtered = [x for x in legal_actions if x != a]
        return filtered or legal_actions

    return mask_fn


def train_base(seed):
    rng = np.random.default_rng(seed)
    env = MaskedLeducPoker(LeducPokerEnv())
    agent = QLearningAgent(num_actions=NUM_ACTIONS, alpha=0.1, epsilon=0.15)
    for _ in range(PRETRAIN):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)
    return env, agent, rng


def learn_attack(seed, agent):
    rng = np.random.default_rng(seed + 1000)
    env = MaskedLeducPoker(LeducPokerEnv())
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


def train_mask_ensemble(seed):
    env, agent, rng = train_base(seed)
    states = collect_states(agent, rng)
    masks = [make_persistent_random_mask(states, seed * 100 + i) for i in range(8)]
    for ep in range(DEFENSE_EPISODES):
        env.set_mask(masks[ep % len(masks)])
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)
    env.set_mask(None)
    return env, agent, rng


def main():
    print("Mask-ensemble defense (Leduc QL)", flush=True)
    print("=" * 60, flush=True)
    results = {"standard": [], "dropout": [], "mask_ensemble": []}

    for seed in SEEDS:
        # Standard victim.
        env_s, agent_s, rng_s = train_base(seed)
        adv_s = learn_attack(seed, agent_s)
        results["standard"].append(evaluate_agent(env_s, agent_s, rng_s, EVAL, mask_fn=adv_s.mask_fn))

        # Independent stochastic action-dropout during training.
        env_d, agent_d, rng_d = train_base(seed)
        states_d = collect_states(agent_d, rng_d)
        dropout = make_persistent_random_mask(states_d, seed + 300, budget_fraction=0.2)
        env_d.set_mask(dropout)
        for _ in range(DEFENSE_EPISODES):
            r, t = play_episode(env_d, agent_d, rng_d)
            agent_d.update(t, r)
        env_d.set_mask(None)
        adv_d = learn_attack(seed + 10, agent_d)
        results["dropout"].append(evaluate_agent(env_d, agent_d, rng_d, EVAL, mask_fn=adv_d.mask_fn))

        # Persistent mask ensemble defense.
        env_e, agent_e, rng_e = train_mask_ensemble(seed)
        adv_e = learn_attack(seed + 20, agent_e)
        results["mask_ensemble"].append(evaluate_agent(env_e, agent_e, rng_e, EVAL, mask_fn=adv_e.mask_fn))
        print(f"  seed={seed} done", flush=True)

    for name, vals in results.items():
        vals = np.asarray(vals, dtype=float)
        mean = vals.mean()
        ci = 1.96 * vals.std() / np.sqrt(len(vals))
        print(f"  {name:>14s}: {mean:+.3f} +/- {ci:.3f}", flush=True)


if __name__ == "__main__":
    main()
