"""Separate-networks ablation: does shared-parameter self-play create artifacts?

Compare:
1. Shared agent (current setup): one agent plays both sides
2. Two separate agents: independent Q-tables for P0 and P1
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.envs.kuhn_poker import KuhnPokerEnv, MaskedKuhnPoker
from core.agents.q_learning import QLearningAgent
from adversary.masking_policy import AdversarialMask

SEEDS = list(range(5))
PRETRAIN = 10000
ATTACK_EPS = 10000
EVAL_EPS = 3000


def play_episode_separate(env, agent_p0, agent_p1, rng):
    """Self-play with two separate agents."""
    env.reset(rng=rng)
    trajectory = []
    while not env.is_terminal:
        player = env.current_player
        info = env.info_state(player)
        legal = env.legal_actions()
        agent = agent_p0 if player == 0 else agent_p1
        action = agent.select_action(info, legal, rng)
        trajectory.append((player, info, action))
        env.step(action)
    return env.returns[0], trajectory


def play_episode_shared(env, agent, rng):
    """Self-play with one shared agent."""
    env.reset(rng=rng)
    trajectory = []
    while not env.is_terminal:
        player = env.current_player
        info = env.info_state(player)
        legal = env.legal_actions()
        action = agent.select_action(info, legal, rng)
        trajectory.append((player, info, action))
        env.step(action)
    return env.returns[0], trajectory


def evaluate(env, agent_or_pair, rng, n, mask_fn=None, separate=False):
    if mask_fn:
        env.set_mask(mask_fn)
    else:
        env.set_mask(None)
    rewards = []
    for _ in range(n):
        if separate:
            r, _ = play_episode_separate(env, agent_or_pair[0], agent_or_pair[1], rng)
        else:
            r, _ = play_episode_shared(env, agent_or_pair, rng)
        rewards.append(r)
    env.set_mask(None)
    return float(np.mean(rewards))


def run_shared(seed):
    rng = np.random.default_rng(seed)
    env = MaskedKuhnPoker(KuhnPokerEnv())
    agent = QLearningAgent(num_actions=2, alpha=0.1, epsilon=0.15)
    for _ in range(PRETRAIN):
        r, t = play_episode_shared(env, agent, rng)
        agent.update(t, r)
    pre = evaluate(env, agent, rng, EVAL_EPS)

    adv = AdversarialMask(target_player=0, num_actions=2, lr=0.01)
    env.set_mask(adv.mask_fn)
    for _ in range(20):
        batch = []
        for _ in range(500):
            r, t = play_episode_shared(env, agent, rng)
            agent.update(t, r)
            batch.append((t, r))
        adv.update(batch)
    post = evaluate(env, agent, rng, EVAL_EPS, mask_fn=adv.mask_fn)
    return pre, post


def run_separate(seed):
    rng = np.random.default_rng(seed)
    env = MaskedKuhnPoker(KuhnPokerEnv())
    p0 = QLearningAgent(num_actions=2, alpha=0.1, epsilon=0.15)
    p1 = QLearningAgent(num_actions=2, alpha=0.1, epsilon=0.15)

    for _ in range(PRETRAIN):
        r, t = play_episode_separate(env, p0, p1, rng)
        for player, info, action in t:
            reward = r if player == 0 else -r
            agent = p0 if player == 0 else p1
            agent.q[info][action] += agent.alpha * (reward - agent.q[info][action])

    pre = evaluate(env, (p0, p1), rng, EVAL_EPS, separate=True)

    adv = AdversarialMask(target_player=0, num_actions=2, lr=0.01)
    env.set_mask(adv.mask_fn)
    for _ in range(20):
        batch = []
        for _ in range(500):
            r, t = play_episode_separate(env, p0, p1, rng)
            for player, info, action in t:
                reward = r if player == 0 else -r
                agent = p0 if player == 0 else p1
                agent.q[info][action] += agent.alpha * (reward - agent.q[info][action])
            batch.append((t, r))
        adv.update(batch)
    post = evaluate(env, (p0, p1), rng, EVAL_EPS, mask_fn=adv.mask_fn, separate=True)
    return pre, post


def main():
    print("=" * 60, flush=True)
    print("Shared vs Separate Networks Ablation (Kuhn QL, 5 seeds)", flush=True)
    print("=" * 60, flush=True)

    for mode, runner in [("Shared", run_shared), ("Separate", run_separate)]:
        pre_vals, post_vals = [], []
        for seed in SEEDS:
            pre, post = runner(seed)
            pre_vals.append(pre)
            post_vals.append(post)
        mp = np.mean(pre_vals)
        mpo = np.mean(post_vals)
        ci = 1.96 * np.std(post_vals) / np.sqrt(len(post_vals))
        print(f"  {mode:>10s}: pre={mp:+.4f}  post={mpo:+.4f} +/-{ci:.4f}  "
              f"delta={mpo-mp:+.4f}", flush=True)


if __name__ == "__main__":
    main()
