"""Causal CACv intervention experiment.

Train compressed vs. redundancy-regularized victims, then attack both with a
matched learned adversary. The goal is to test whether directly manipulating
Q-value gaps changes attack success.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adversary.mask_utils import evaluate_agent
from adversary.masking_policy import AdversarialMask
from core.agents.q_learning import QLearningAgent
from core.envs.kuhn_poker import KuhnPokerEnv, MaskedKuhnPoker
from core.envs.leduc_poker import LeducPokerEnv, MaskedLeducPoker, NUM_ACTIONS as LEDUC_ACTIONS
from core.training.selfplay import play_episode
from experiments.cacv_regularized_defense import estimate_cacv_targets, make_env


SEEDS = list(range(5))
TRAIN_EPISODES = 12000
INTERVENTION_EPISODES = 6000
ATTACK_OUTER = 12
ATTACK_INNER = 400
EVAL = 3000


def train_victim(game, seed, redundant=False):
    rng = np.random.default_rng(seed)
    env, num_actions = make_env(game)
    agent = QLearningAgent(
        num_actions=num_actions,
        alpha=0.1,
        epsilon=0.15,
        regularize=redundant,
        reg_tau=0.5,
        reg_lambda=0.015,
    )
    for ep in range(TRAIN_EPISODES + INTERVENTION_EPISODES):
        if redundant and ep >= TRAIN_EPISODES and (ep - TRAIN_EPISODES) % 1000 == 0:
            targets, _ = estimate_cacv_targets(agent, game, rng, top_k=12 if game == "leduc" else 3)
            agent.set_regularization_states(targets)
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)
    return env, agent, rng


def state_gap_stats(agent, game, rng):
    _, scores = estimate_cacv_targets(agent, game, rng, top_k=100, rollouts=1000)
    gaps = []
    for state in scores:
        q = agent.q[state]
        gaps.append(float(q.max() - q.min()))
    gaps = np.asarray(gaps, dtype=float)
    return {
        "mean_gap": float(gaps.mean()) if len(gaps) else 0.0,
        "p90_gap": float(np.quantile(gaps, 0.9)) if len(gaps) else 0.0,
        "top3_share": float(sum(sorted(scores.values(), reverse=True)[:3]) / sum(scores.values()))
        if sum(scores.values()) > 0
        else 0.0,
    }


def learn_attack(game, seed, agent):
    rng = np.random.default_rng(seed + 20_000)
    env, num_actions = make_env(game)
    adv = AdversarialMask(target_player=0, num_actions=num_actions, lr=0.01)
    env.set_mask(adv.mask_fn)
    for _ in range(ATTACK_OUTER):
        batch = []
        for _ in range(ATTACK_INNER):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)
            batch.append((t, r))
        adv.update(batch)
    return adv


def run_one(game, seed, redundant):
    env, agent, rng = train_victim(game, seed, redundant=redundant)
    clean = evaluate_agent(env, agent, rng, EVAL)
    stats = state_gap_stats(agent, game, rng)
    adv = learn_attack(game, seed, agent)
    attacked = evaluate_agent(env, agent, rng, EVAL, mask_fn=adv.mask_fn)
    return clean, attacked, stats["mean_gap"], stats["p90_gap"], stats["top3_share"]


def main():
    print("CACv causal intervention: compressed vs redundant victims", flush=True)
    for game in ["kuhn", "leduc"]:
        print(f"\n== {game.upper()} ==", flush=True)
        results = {"compressed": [], "redundant": []}
        for mode in results:
            redundant = mode == "redundant"
            for seed in SEEDS:
                vals = run_one(game, seed, redundant)
                results[mode].append(vals)
                clean, attacked, mean_gap, p90_gap, top3 = vals
                print(
                    f"{mode:>10s} seed={seed} clean={clean:+.3f} attacked={attacked:+.3f} "
                    f"mean_gap={mean_gap:.3f} p90_gap={p90_gap:.3f} top3={top3:.3f}",
                    flush=True,
                )
        for mode, vals in results.items():
            arr = np.asarray(vals, dtype=float)
            means = arr.mean(axis=0)
            cis = 1.96 * arr.std(axis=0) / np.sqrt(len(arr))
            print(
                f"{mode:>10s}: clean={means[0]:+.3f}+/-{cis[0]:.3f}, "
                f"attacked={means[1]:+.3f}+/-{cis[1]:.3f}, "
                f"mean_gap={means[2]:.3f}, p90_gap={means[3]:.3f}, top3={means[4]:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
