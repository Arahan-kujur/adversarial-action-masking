"""Adversarial co-training defense.

Periodically retrain a learned adversary against the current victim, then
let the victim train under that adversary's mask. This closes the loop:
the defense is targeted at exactly the states an actual adversary would
attack.
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


SEEDS = [0, 1, 2, 3, 4]
PRETRAIN = 6000
COTRAIN_PHASES = 4
ADV_TRAIN_OUTER = 6
ADV_TRAIN_INNER = 200
DEFENSE_INNER = 1500
EVAL = 2000
FINAL_ATTACK_OUTER = 12
FINAL_ATTACK_INNER = 400


def make_env(game):
    if game == "kuhn":
        return MaskedKuhnPoker(KuhnPokerEnv()), 2
    return MaskedLeducPoker(LeducPokerEnv()), LEDUC_ACTIONS


def train(agent, env, rng, episodes, mask_fn=None):
    if mask_fn is not None:
        env.set_mask(mask_fn)
    else:
        env.set_mask(None)
    for _ in range(episodes):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)


def train_attacker(game, seed, agent, outer, inner):
    rng = np.random.default_rng(seed + 9000)
    env, num_actions = make_env(game)
    adv = AdversarialMask(target_player=0, num_actions=num_actions, lr=0.01)
    env.set_mask(adv.mask_fn)
    for _ in range(outer):
        batch = []
        for _ in range(inner):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)
            batch.append((t, r))
        adv.update(batch)
    env.set_mask(None)
    return adv


def cotrain_victim(game, seed):
    rng = np.random.default_rng(seed)
    env, num_actions = make_env(game)
    agent = QLearningAgent(num_actions=num_actions, alpha=0.1, epsilon=0.15)
    train(agent, env, rng, PRETRAIN)
    for _ in range(COTRAIN_PHASES):
        adv = train_attacker(game, seed, agent, ADV_TRAIN_OUTER, ADV_TRAIN_INNER)
        train(agent, env, rng, DEFENSE_INNER, mask_fn=adv.mask_fn)
    return env, agent, rng


def standard_victim(game, seed):
    rng = np.random.default_rng(seed)
    env, num_actions = make_env(game)
    agent = QLearningAgent(num_actions=num_actions, alpha=0.1, epsilon=0.15)
    total = PRETRAIN + COTRAIN_PHASES * DEFENSE_INNER
    train(agent, env, rng, total)
    return env, agent, rng


def evaluate_under_attack(game, seed, agent, env, rng):
    rng_eval = np.random.default_rng(seed + 12345)
    env_eval, num_actions = make_env(game)
    adv = AdversarialMask(target_player=0, num_actions=num_actions, lr=0.01)
    env_eval.set_mask(adv.mask_fn)
    for _ in range(FINAL_ATTACK_OUTER):
        batch = []
        for _ in range(FINAL_ATTACK_INNER):
            r, t = play_episode(env_eval, agent, rng_eval)
            agent.update(t, r)
            batch.append((t, r))
        adv.update(batch)
    return evaluate_agent(env_eval, agent, rng_eval, EVAL, mask_fn=adv.mask_fn)


def run(game):
    print(f"\n== {game.upper()} ==", flush=True)
    rows = {"standard": [], "cotrained": []}
    for seed in SEEDS:
        env_s, agent_s, rng_s = standard_victim(game, seed)
        clean_s = evaluate_agent(env_s, agent_s, rng_s, EVAL)
        atk_s = evaluate_under_attack(game, seed, agent_s, env_s, rng_s)
        rows["standard"].append((clean_s, atk_s))
        print(
            f"  standard  seed={seed} clean={clean_s:+.3f} attacked={atk_s:+.3f}",
            flush=True,
        )

        env_c, agent_c, rng_c = cotrain_victim(game, seed)
        clean_c = evaluate_agent(env_c, agent_c, rng_c, EVAL)
        atk_c = evaluate_under_attack(game, seed, agent_c, env_c, rng_c)
        rows["cotrained"].append((clean_c, atk_c))
        print(
            f"  cotrained seed={seed} clean={clean_c:+.3f} attacked={atk_c:+.3f}",
            flush=True,
        )
    for name, vals in rows.items():
        arr = np.asarray(vals, dtype=float)
        m = arr.mean(axis=0)
        ci = 1.96 * arr.std(axis=0) / np.sqrt(len(arr))
        print(
            f"  {name:>9s}: clean={m[0]:+.3f}+/-{ci[0]:.3f} attacked={m[1]:+.3f}+/-{ci[1]:.3f}",
            flush=True,
        )


def main():
    print("Adversarial co-training defense", flush=True)
    for game in ["kuhn", "leduc"]:
        run(game)


if __name__ == "__main__":
    main()
