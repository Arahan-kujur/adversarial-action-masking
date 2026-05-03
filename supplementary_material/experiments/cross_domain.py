"""Cross-domain validation: adversarial action masking in Competitive Gridworld.

Shows the phenomenon generalises beyond poker to a completely different
game type (spatial, perfect information, different reward structure).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.envs.gridworld import CompetitiveGridworld, MaskedGridworld, NUM_ACTIONS, UP
from core.agents.q_learning import QLearningAgent
from core.training.selfplay import play_episode
from adversary.masking_policy import AdversarialMask, random_mask, fixed_removal
from adversary.mask_utils import evaluate_agent

SEEDS = list(range(5))
PRETRAIN = 20000
ADV_OUTER = 25
ADV_INNER = 500
CONTINUED = 10000
EVAL_EPS = 3000
WINDOW = 500


def run_one(mode, seed):
    rng = np.random.default_rng(seed)
    env = MaskedGridworld(CompetitiveGridworld())
    agent = QLearningAgent(num_actions=NUM_ACTIONS, alpha=0.1, epsilon=0.2)

    # Pretrain
    for _ in range(PRETRAIN):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)

    pre = evaluate_agent(env, agent, rng, EVAL_EPS)

    if mode == "none":
        return pre, pre

    if mode == "random":
        rm = random_mask(0, 0.3, np.random.default_rng(seed + 99))
        env.set_mask(rm)
        curve = []
        for i in range(ADV_OUTER * ADV_INNER + CONTINUED):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)
            curve.append(r)
        post = evaluate_agent(env, agent, rng, EVAL_EPS, mask_fn=rm)
        return pre, post

    if mode == "fixed":
        fm = fixed_removal(0, UP)
        env.set_mask(fm)
        for _ in range(ADV_OUTER * ADV_INNER + CONTINUED):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)
        post = evaluate_agent(env, agent, rng, EVAL_EPS, mask_fn=fm)
        return pre, post

    if mode == "adversarial":
        adv = AdversarialMask(target_player=0, num_actions=NUM_ACTIONS, lr=0.01)
        env.set_mask(adv.mask_fn)
        for outer in range(ADV_OUTER):
            batch = []
            for _ in range(ADV_INNER):
                r, t = play_episode(env, agent, rng)
                agent.update(t, r)
                batch.append((t, r))
            adv.update(batch)

        # Continued training under converged mask
        for _ in range(CONTINUED):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)

        post = evaluate_agent(env, agent, rng, EVAL_EPS, mask_fn=adv.mask_fn)
        return pre, post


def main():
    print("=" * 65, flush=True)
    print("  CROSS-DOMAIN: Competitive Gridworld (5x5)", flush=True)
    print("  P0=prey (reach goal), P1=predator (catch prey)", flush=True)
    print("  5 actions: UP/DOWN/LEFT/RIGHT/STAY", flush=True)
    print("=" * 65, flush=True)

    modes = ["none", "random", "fixed", "adversarial"]

    for mode in modes:
        pre_vals, post_vals = [], []
        for seed in SEEDS:
            pre, post = run_one(mode, seed)
            pre_vals.append(pre)
            post_vals.append(post)

        mp = np.mean(pre_vals)
        mpo = np.mean(post_vals)
        ci = 1.96 * np.std(post_vals) / np.sqrt(len(post_vals))
        delta = mpo - mp
        print(f"  {mode:>12s}: pre={mp:+.3f}  post={mpo:+.3f} +/-{ci:.3f}  "
              f"delta={delta:+.3f}", flush=True)

    print(flush=True)

    # Count info states
    rng = np.random.default_rng(0)
    env = MaskedGridworld(CompetitiveGridworld())
    agent = QLearningAgent(num_actions=NUM_ACTIONS, alpha=0.1, epsilon=0.2)
    states = set()
    for _ in range(50000):
        env.reset(rng=rng)
        while not env.is_terminal:
            p = env.current_player
            info = env.info_state(p)
            if p == 0:
                states.add(info)
            legal = env.legal_actions()
            a = agent.select_action(info, legal, rng)
            env.step(a)
    print(f"  Unique P0 info states observed: {len(states)}", flush=True)


if __name__ == "__main__":
    main()
