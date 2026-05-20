"""Count P0 information states reachable under uniform random play.

This isolates game-tree size from policy concentration, addressing the
reviewer's concern that observed state counts depend on pretraining.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.envs.kuhn_poker import KuhnPokerEnv, MaskedKuhnPoker
from core.envs.leduc_poker import LeducPokerEnv, MaskedLeducPoker
from core.envs.leduc_n import LeducNPokerEnv, MaskedLeducNPoker


def make_env(name):
    if name == "Kuhn":
        return MaskedKuhnPoker(KuhnPokerEnv())
    if name == "Leduc":
        return MaskedLeducPoker(LeducPokerEnv())
    n = int(name.split("-")[1])
    return MaskedLeducNPoker(LeducNPokerEnv(num_ranks=n))


def count_uniform_random(env, episodes, rng):
    states = set()
    for _ in range(episodes):
        env.reset(rng=rng)
        while not env.is_terminal:
            p = env.current_player
            info = env.info_state(p)
            legal = env.legal_actions()
            if not legal:
                break
            if p == 0:
                states.add(info)
            action = int(rng.choice(legal))
            env.step(action)
    return len(states)


def main():
    print("Uniform-random P0 information state counts", flush=True)
    print(f"{'Game':>10s} {'Episodes':>10s} {'Unique P0 states':>18s}", flush=True)
    configs = [
        ("Kuhn", 5000),
        ("Leduc", 50000),
        ("Leduc-5", 100000),
        ("Leduc-10", 100000),
        ("Leduc-20", 100000),
        ("Leduc-30", 100000),
        ("Leduc-50", 100000),
    ]
    rng = np.random.default_rng(0)
    for name, episodes in configs:
        env = make_env(name)
        n = count_uniform_random(env, episodes, rng)
        print(f"{name:>10s} {episodes:>10d} {n:>18d}", flush=True)


if __name__ == "__main__":
    main()
