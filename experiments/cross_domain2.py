"""Cross-domain #2: Resource Collection Game.

Competitive resource gathering on a 4x4 grid. Different structure
from both poker (no cards/betting) and gridworld (no single goal).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.envs.resource_collection import (ResourceCollectionEnv,
    MaskedResourceCollection, NUM_ACTIONS)
from core.agents.q_learning import QLearningAgent
from core.training.selfplay import play_episode
from adversary.masking_policy import AdversarialMask, random_mask, fixed_removal
from adversary.mask_utils import evaluate_agent

SEEDS = list(range(5))
PRETRAIN = 20000
ADV_OUTER, ADV_INNER = 25, 500
EVAL = 3000


def run_one(mode, seed):
    rng = np.random.default_rng(seed)
    env = MaskedResourceCollection(ResourceCollectionEnv())
    agent = QLearningAgent(num_actions=NUM_ACTIONS, alpha=0.1, epsilon=0.2)

    for _ in range(PRETRAIN):
        r, t = play_episode(env, agent, rng); agent.update(t, r)

    pre = evaluate_agent(env, agent, rng, EVAL)

    if mode == "none":
        return pre, pre

    if mode == "random":
        rm = random_mask(0, 0.3, np.random.default_rng(seed + 99))
        env.set_mask(rm)
        for _ in range(ADV_OUTER * ADV_INNER):
            r, t = play_episode(env, agent, rng); agent.update(t, r)
        return pre, evaluate_agent(env, agent, rng, EVAL, mask_fn=rm)

    if mode == "fixed":
        from core.envs.resource_collection import UP
        fm = fixed_removal(0, UP)
        env.set_mask(fm)
        for _ in range(ADV_OUTER * ADV_INNER):
            r, t = play_episode(env, agent, rng); agent.update(t, r)
        return pre, evaluate_agent(env, agent, rng, EVAL, mask_fn=fm)

    if mode == "adversarial":
        adv = AdversarialMask(target_player=0, num_actions=NUM_ACTIONS, lr=0.01)
        env.set_mask(adv.mask_fn)
        for o in range(ADV_OUTER):
            batch = []
            for _ in range(ADV_INNER):
                r, t = play_episode(env, agent, rng); agent.update(t, r)
                batch.append((t, r))
            adv.update(batch)
        return pre, evaluate_agent(env, agent, rng, EVAL, mask_fn=adv.mask_fn)


def main():
    print("="*65, flush=True)
    print("  CROSS-DOMAIN #2: Resource Collection (4x4, 4 actions)", flush=True)
    print("="*65, flush=True)

    for mode in ["none", "random", "fixed", "adversarial"]:
        pre_v, post_v = [], []
        for seed in SEEDS:
            pre, post = run_one(mode, seed)
            pre_v.append(pre); post_v.append(post)
        mp, mpo = np.mean(pre_v), np.mean(post_v)
        ci = 1.96 * np.std(post_v) / len(post_v)**.5
        print(f"  {mode:>12s}: pre={mp:+.3f}  post={mpo:+.3f} +/-{ci:.3f}  "
              f"delta={mpo-mp:+.3f}", flush=True)

    # Count states
    rng = np.random.default_rng(0)
    env = MaskedResourceCollection(ResourceCollectionEnv())
    agent = QLearningAgent(num_actions=NUM_ACTIONS, alpha=0.1, epsilon=0.2)
    states = set()
    for _ in range(50000):
        env.reset(rng=rng)
        while not env.is_terminal:
            p = env.current_player; info = env.info_state(p)
            if p == 0: states.add(info)
            a = agent.select_action(info, env.legal_actions(), rng); env.step(a)
    print(f"\n  Unique P0 info states: {len(states)}", flush=True)

if __name__ == "__main__":
    main()
