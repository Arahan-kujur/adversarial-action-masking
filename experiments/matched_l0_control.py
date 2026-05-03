"""Strict matched-L0 random baseline for Leduc.

The adversarial mask first determines the number of information states it
actually masks. The matched-random baseline then masks exactly the same number
of states, sampled from the same observed state set, so coverage is matched and
only targeting differs.
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adversary.masking_policy import AdversarialMask
from adversary.mask_utils import evaluate_agent
from core.agents.q_learning import QLearningAgent
from core.envs.leduc_poker import LeducPokerEnv, MaskedLeducPoker, NUM_ACTIONS
from core.training.selfplay import play_episode


SEEDS = list(range(5))


def train_base(seed):
    rng = np.random.default_rng(seed)
    env = MaskedLeducPoker(LeducPokerEnv())
    agent = QLearningAgent(num_actions=NUM_ACTIONS, alpha=0.1, epsilon=0.15)
    for _ in range(12000):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)
    return env, agent, rng


def observed_p0_states(agent, rng, episodes=3000):
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


def effective_masked_states(agent, mask_fn, rng, episodes=3000):
    env = MaskedLeducPoker(LeducPokerEnv())
    masked = set()
    for _ in range(episodes):
        env.reset(rng=rng)
        while not env.is_terminal:
            p = env.current_player
            info = env.info_state(p)
            base = env.env.legal_actions()
            if p == 0 and len(mask_fn(info, base, p)) < len(base):
                masked.add(info)
            a = agent.select_action(info, env.legal_actions(), rng)
            env.step(a)
    return masked


def matched_random_mask(target_states, seed, num_actions=NUM_ACTIONS):
    rng = np.random.default_rng(seed)
    # Fix one removed action per selected state for reproducibility.
    removed = {s: int(rng.integers(0, num_actions)) for s in target_states}

    def mask_fn(info_state, legal_actions, player):
        if player != 0 or info_state not in removed or len(legal_actions) <= 1:
            return legal_actions
        candidate = removed[info_state]
        if candidate not in legal_actions:
            candidate = legal_actions[-1]
        filtered = [a for a in legal_actions if a != candidate]
        return filtered or legal_actions

    return mask_fn


def main():
    print("Strict matched-L0 random control (Leduc QL)", flush=True)
    print("=" * 60, flush=True)
    adv_vals, matched_vals, random_k_vals = [], [], []
    for seed in SEEDS:
        env, agent, rng = train_base(seed)

        adv_agent = copy.deepcopy(agent)
        adv_env = MaskedLeducPoker(LeducPokerEnv())
        adv = AdversarialMask(target_player=0, num_actions=NUM_ACTIONS, lr=0.01)
        adv_env.set_mask(adv.mask_fn)
        for _ in range(15):
            batch = []
            for _ in range(400):
                r, t = play_episode(adv_env, adv_agent, rng)
                adv_agent.update(t, r)
                batch.append((t, r))
            adv.update(batch)

        adv_masked = effective_masked_states(adv_agent, adv.mask_fn, rng)
        k = len(adv_masked)
        all_states = observed_p0_states(agent, rng)
        selected = rng.choice(all_states, size=min(k, len(all_states)), replace=False)
        matched = matched_random_mask(selected, seed + 900)

        matched_agent = copy.deepcopy(agent)
        matched_env = MaskedLeducPoker(LeducPokerEnv())
        matched_env.set_mask(matched)
        for _ in range(15 * 400):
            r, t = play_episode(matched_env, matched_agent, rng)
            matched_agent.update(t, r)

        adv_score = evaluate_agent(adv_env, adv_agent, rng, 2000, mask_fn=adv.mask_fn)
        matched_score = evaluate_agent(matched_env, matched_agent, rng, 2000, mask_fn=matched)
        adv_vals.append(adv_score)
        matched_vals.append(matched_score)
        random_k_vals.append(k)
        print(f"seed={seed}: adv_k={k}, adv={adv_score:+.3f}, matched={matched_score:+.3f}", flush=True)

    def fmt(values):
        values = np.asarray(values, dtype=float)
        return values.mean(), 1.96 * values.std() / np.sqrt(len(values))

    adv_m, adv_ci = fmt(adv_vals)
    mat_m, mat_ci = fmt(matched_vals)
    k_m, k_ci = fmt(random_k_vals)
    print(f"\nMatched L0 k: {k_m:.1f} +/- {k_ci:.1f}", flush=True)
    print(f"Adversarial: {adv_m:+.3f} +/- {adv_ci:.3f}", flush=True)
    print(f"Matched random: {mat_m:+.3f} +/- {mat_ci:.3f}", flush=True)
    print(f"Adv/matched-random damage ratio: {(abs(adv_m) / max(abs(mat_m), 1e-6)):.2f}x", flush=True)


if __name__ == "__main__":
    main()
