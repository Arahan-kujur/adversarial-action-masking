"""CACv-regularized self-play defense.

This experiment tests whether preserving multiple near-optimal actions at
high-reach, high-value-gap states improves robustness to action removal.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adversary.mask_utils import evaluate_agent
from adversary.masking_policy import AdversarialMask, random_mask
from core.agents.q_learning import QLearningAgent
from core.envs.kuhn_poker import KuhnPokerEnv, MaskedKuhnPoker
from core.envs.leduc_poker import LeducPokerEnv, MaskedLeducPoker, NUM_ACTIONS as LEDUC_ACTIONS
from core.training.selfplay import play_episode


SEEDS = list(range(5))
PRETRAIN = 8000
DEFENSE_EPISODES = 8000
ATTACK_OUTER = 12
ATTACK_INNER = 400
EVAL = 3000


def make_env(game):
    if game == "kuhn":
        return MaskedKuhnPoker(KuhnPokerEnv()), 2
    if game == "leduc":
        return MaskedLeducPoker(LeducPokerEnv()), LEDUC_ACTIONS
    raise ValueError(game)


def train(agent, env, rng, episodes):
    for _ in range(episodes):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)


def estimate_cacv_targets(agent, game, rng, top_k=12, rollouts=500):
    """Rank visited P0 states by reach * Q-value gap."""
    env, num_actions = make_env(game)
    counts = {}
    for _ in range(rollouts):
        env.reset(rng=rng)
        seen = set()
        while not env.is_terminal:
            p = env.current_player
            info = env.info_state(p)
            legal = env.legal_actions()
            if p == 0:
                seen.add(info)
            action = agent.select_action(info, legal, rng)
            env.step(action)
        for state in seen:
            counts[state] = counts.get(state, 0) + 1

    scores = {}
    for state, count in counts.items():
        q = agent.q[state]
        gap = float(q.max() - q.min())
        scores[state] = (count / max(1, rollouts)) * gap
    ranked = sorted(scores, key=scores.get, reverse=True)
    return set(ranked[: min(top_k, len(ranked))]), scores


def train_cacv_regularized(game, seed):
    rng = np.random.default_rng(seed)
    env, num_actions = make_env(game)
    agent = QLearningAgent(
        num_actions=num_actions,
        alpha=0.1,
        epsilon=0.15,
        regularize=True,
        reg_tau=0.5,
        reg_lambda=0.01,
    )
    train(agent, env, rng, PRETRAIN)

    for ep in range(DEFENSE_EPISODES):
        if ep % 1000 == 0:
            targets, _ = estimate_cacv_targets(agent, game, rng, top_k=12 if game == "leduc" else 3)
            agent.set_regularization_states(targets)
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)
    return env, agent, rng


def train_standard(game, seed):
    rng = np.random.default_rng(seed)
    env, num_actions = make_env(game)
    agent = QLearningAgent(num_actions=num_actions, alpha=0.1, epsilon=0.15)
    train(agent, env, rng, PRETRAIN + DEFENSE_EPISODES)
    return env, agent, rng


def train_dropout(game, seed):
    rng = np.random.default_rng(seed)
    env, num_actions = make_env(game)
    agent = QLearningAgent(num_actions=num_actions, alpha=0.1, epsilon=0.15)
    train(agent, env, rng, PRETRAIN)
    dropout = random_mask(0, 0.2, np.random.default_rng(seed + 777))
    env.set_mask(dropout)
    train(agent, env, rng, DEFENSE_EPISODES)
    env.set_mask(None)
    return env, agent, rng


def learn_attack(game, seed, agent):
    rng = np.random.default_rng(seed + 10_000)
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


def run_condition(game, condition, seed):
    if condition == "standard":
        env, agent, rng = train_standard(game, seed)
    elif condition == "dropout":
        env, agent, rng = train_dropout(game, seed)
    elif condition == "cacv_regularized":
        env, agent, rng = train_cacv_regularized(game, seed)
    else:
        raise ValueError(condition)

    clean = evaluate_agent(env, agent, rng, EVAL)
    adv = learn_attack(game, seed, agent)
    attacked = evaluate_agent(env, agent, rng, EVAL, mask_fn=adv.mask_fn)
    targets, scores = estimate_cacv_targets(agent, game, rng, top_k=100)
    total_score = sum(scores.values())
    top_score = sum(sorted(scores.values(), reverse=True)[: min(3, len(scores))])
    concentration = top_score / total_score if total_score > 0 else 0.0
    return clean, attacked, concentration


def main():
    print("CACv-regularized self-play defense", flush=True)
    for game in ["kuhn", "leduc"]:
        print(f"\n== {game.upper()} ==", flush=True)
        results = {k: [] for k in ["standard", "dropout", "cacv_regularized"]}
        for condition in results:
            for seed in SEEDS:
                clean, attacked, concentration = run_condition(game, condition, seed)
                results[condition].append((clean, attacked, concentration))
                print(
                    f"{condition:>16s} seed={seed} clean={clean:+.3f} "
                    f"attacked={attacked:+.3f} top3={concentration:.3f}",
                    flush=True,
                )
        for condition, vals in results.items():
            arr = np.asarray(vals, dtype=float)
            means = arr.mean(axis=0)
            cis = 1.96 * arr.std(axis=0) / np.sqrt(len(arr))
            print(
                f"{condition:>16s}: clean={means[0]:+.3f}±{cis[0]:.3f}, "
                f"attacked={means[1]:+.3f}±{cis[1]:.3f}, top3={means[2]:.3f}±{cis[2]:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
