"""NFSP victim in Leduc Poker under adversarial masking.

Full-scale experiment: 5 seeds, pre-train then adversarial attack.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.envs.leduc_poker import LeducPokerEnv, MaskedLeducPoker, NUM_ACTIONS
from core.agents.nfsp import NFSPAgent
from core.training.selfplay import play_episode
from adversary.masking_policy import AdversarialMask, random_mask, fixed_removal
from adversary.mask_utils import evaluate_agent

SEEDS = list(range(5))
PRETRAIN = 15000
ATTACK_EPS = 10000
EVAL_EPS = 3000
ADV_OUTER = 20
ADV_INNER = 500


def run_one(mode, seed):
    rng = np.random.default_rng(seed)
    env = MaskedLeducPoker(LeducPokerEnv())
    agent = NFSPAgent(num_actions=NUM_ACTIONS, alpha=0.1, epsilon=0.15, eta=0.1)

    for i in range(PRETRAIN):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)

    pre_reward = evaluate_agent(env, agent, rng, EVAL_EPS)

    if mode == "none":
        return pre_reward, pre_reward

    if mode == "random":
        rm = random_mask(target_player=0, remove_prob=0.5,
                         rng=np.random.default_rng(seed + 99))
        env.set_mask(rm)
        for _ in range(ATTACK_EPS):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)
        post = evaluate_agent(env, agent, rng, EVAL_EPS, mask_fn=rm)
        return pre_reward, post

    if mode == "fixed":
        fm = fixed_removal(target_player=0, removed_action=2)  # remove RAISE
        env.set_mask(fm)
        for _ in range(ATTACK_EPS):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)
        post = evaluate_agent(env, agent, rng, EVAL_EPS, mask_fn=fm)
        return pre_reward, post

    if mode == "adversarial":
        adv = AdversarialMask(target_player=0, num_actions=NUM_ACTIONS, lr=0.01)
        env.set_mask(adv.mask_fn)
        for o in range(ADV_OUTER):
            batch = []
            for _ in range(ADV_INNER):
                r, t = play_episode(env, agent, rng)
                agent.update(t, r)
                batch.append((t, r))
            adv.update(batch)
        post = evaluate_agent(env, agent, rng, EVAL_EPS, mask_fn=adv.mask_fn)
        return pre_reward, post


def main():
    print("=" * 60, flush=True)
    print("NFSP Victim in LEDUC POKER Under Adversarial Masking", flush=True)
    print(f"Seeds: {len(SEEDS)}, Pretrain: {PRETRAIN}, Attack: {ATTACK_EPS}", flush=True)
    print("=" * 60, flush=True)

    modes = ["none", "random", "fixed", "adversarial"]

    for mode in modes:
        pre_vals, post_vals = [], []
        for seed in SEEDS:
            print(f"  {mode} seed={seed}...", end="", flush=True)
            pre, post = run_one(mode, seed)
            pre_vals.append(pre)
            post_vals.append(post)
            print(f" pre={pre:+.3f} post={post:+.3f}", flush=True)

        m_pre = np.mean(pre_vals)
        m_post = np.mean(post_vals)
        ci = 1.96 * np.std(post_vals) / np.sqrt(len(post_vals))
        delta = m_post - m_pre
        print(f"  >> {mode:>12s}: pre={m_pre:+.3f}  post={m_post:+.3f} +/-{ci:.3f}  "
              f"delta={delta:+.3f}", flush=True)
        print(flush=True)


if __name__ == "__main__":
    main()
