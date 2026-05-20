"""Robustness frontier: sweep regularization strength and measure
clean reward vs adversarial reward to trace the exploitation-robustness
trade-off curve.
"""
from __future__ import annotations

import json
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
from experiments.cacv_regularized_defense import estimate_cacv_targets


SEEDS = [0, 1, 2]
PRETRAIN = 6000
DEFENSE_EPISODES = 6000
ATTACK_OUTER = 10
ATTACK_INNER = 300
EVAL = 2000


def make_env(game):
    if game == "kuhn":
        return MaskedKuhnPoker(KuhnPokerEnv()), 2
    return MaskedLeducPoker(LeducPokerEnv()), LEDUC_ACTIONS


def train(agent, env, rng, episodes):
    for _ in range(episodes):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)


def train_with_lambda(game, seed, lam):
    rng = np.random.default_rng(seed)
    env, num_actions = make_env(game)
    agent = QLearningAgent(
        num_actions=num_actions,
        alpha=0.1,
        epsilon=0.15,
        regularize=lam > 0,
        reg_tau=0.5,
        reg_lambda=lam,
    )
    train(agent, env, rng, PRETRAIN)
    if lam > 0:
        for ep in range(DEFENSE_EPISODES):
            if ep % 1000 == 0:
                top_k = 12 if game == "leduc" else 3
                targets, _ = estimate_cacv_targets(agent, game, rng, top_k=top_k)
                agent.set_regularization_states(targets)
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)
    else:
        train(agent, env, rng, DEFENSE_EPISODES)
    return env, agent, rng


def learn_attack(game, seed, agent):
    rng = np.random.default_rng(seed + 5000)
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


def run(game, lam, seed):
    env, agent, rng = train_with_lambda(game, seed, lam)
    clean = evaluate_agent(env, agent, rng, EVAL)
    adv = learn_attack(game, seed, agent)
    attacked = evaluate_agent(env, agent, rng, EVAL, mask_fn=adv.mask_fn)
    return clean, attacked


def main():
    lambdas = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]
    out = {"kuhn": [], "leduc": []}
    for game in ["kuhn", "leduc"]:
        print(f"\n== {game.upper()} ==", flush=True)
        for lam in lambdas:
            cleans, attacks = [], []
            for seed in SEEDS:
                c, a = run(game, lam, seed)
                cleans.append(c)
                attacks.append(a)
                print(
                    f"  lambda={lam:.3f} seed={seed} clean={c:+.3f} attacked={a:+.3f}",
                    flush=True,
                )
            out[game].append(
                {
                    "lambda": lam,
                    "clean_mean": float(np.mean(cleans)),
                    "clean_ci": float(1.96 * np.std(cleans) / np.sqrt(len(cleans))),
                    "attacked_mean": float(np.mean(attacks)),
                    "attacked_ci": float(1.96 * np.std(attacks) / np.sqrt(len(attacks))),
                }
            )

    target = Path(__file__).resolve().parents[1] / "paper" / "latex" / "figures"
    target.mkdir(parents=True, exist_ok=True)
    with open(target / "robustness_frontier.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("\nSaved", target / "robustness_frontier.json", flush=True)


if __name__ == "__main__":
    main()
