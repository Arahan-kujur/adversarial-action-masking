"""NFSP on Leduc-5 (389 info states): full experiment with learning curves.

Proper NFSP: replay buffers for both BR and average, longer training.
Reports: pre/post reward, learning curves in 500-ep windows, no-recovery evidence.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.envs.leduc_n import LeducNPokerEnv, MaskedLeducNPoker, NUM_ACTIONS
from core.agents.nfsp import NFSPAgent
from core.training.selfplay import play_episode
from adversary.masking_policy import AdversarialMask, random_mask, fixed_removal
from adversary.mask_utils import evaluate_agent

SEEDS = list(range(5))
PRETRAIN = 20000
ATTACK_OUTER = 25
ATTACK_INNER = 500
CONTINUED = 15000  # continued training under mask after adversary converges
EVAL_EPS = 3000
WINDOW = 500


def run_one(seed):
    rng = np.random.default_rng(seed)
    env = MaskedLeducNPoker(LeducNPokerEnv(num_ranks=5))
    agent = NFSPAgent(num_actions=NUM_ACTIONS, alpha=0.05, epsilon=0.15, eta=0.1)

    # Pretrain
    pre_curve = []
    for i in range(PRETRAIN):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)
        pre_curve.append(r)
        if (i + 1) % 5000 == 0:
            w = pre_curve[-WINDOW:]
            print(f"    pretrain {i+1}/{PRETRAIN}: {np.mean(w):+.3f}", flush=True)

    pre_reward = evaluate_agent(env, agent, rng, EVAL_EPS)
    print(f"    pre-attack eval: {pre_reward:+.4f}", flush=True)

    # Adversarial attack
    adv = AdversarialMask(target_player=0, num_actions=NUM_ACTIONS, lr=0.01)
    env.set_mask(adv.mask_fn)

    attack_curve = []
    for outer in range(ATTACK_OUTER):
        batch = []
        for _ in range(ATTACK_INNER):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)
            batch.append((t, r))
            attack_curve.append(r)
        adv.update(batch)
        if (outer + 1) % 5 == 0:
            w = attack_curve[-WINDOW:]
            print(f"    attack outer {outer+1}/{ATTACK_OUTER}: {np.mean(w):+.3f}",
                  flush=True)

    # Continued training (no-recovery test)
    for i in range(CONTINUED):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)
        attack_curve.append(r)
        if (i + 1) % 5000 == 0:
            w = attack_curve[-WINDOW:]
            print(f"    continued {i+1}/{CONTINUED}: {np.mean(w):+.3f}", flush=True)

    post_reward = evaluate_agent(env, agent, rng, EVAL_EPS, mask_fn=adv.mask_fn)
    print(f"    post-attack eval: {post_reward:+.4f}", flush=True)

    # Random baseline
    env2 = MaskedLeducNPoker(LeducNPokerEnv(num_ranks=5))
    agent2 = NFSPAgent(num_actions=NUM_ACTIONS, alpha=0.05, epsilon=0.15, eta=0.1)
    for _ in range(PRETRAIN):
        r, t = play_episode(env2, agent2, rng)
        agent2.update(t, r)
    rm = random_mask(0, 0.5, np.random.default_rng(seed + 99))
    env2.set_mask(rm)
    for _ in range(ATTACK_OUTER * ATTACK_INNER + CONTINUED):
        r, t = play_episode(env2, agent2, rng)
        agent2.update(t, r)
    rand_reward = evaluate_agent(env2, agent2, rng, EVAL_EPS, mask_fn=rm)

    # Learning curve summary
    curve_windows = []
    all_curve = pre_curve + attack_curve
    for i in range(0, len(all_curve), WINDOW):
        chunk = all_curve[i:i+WINDOW]
        if len(chunk) >= WINDOW // 2:
            curve_windows.append((i + len(chunk)//2, np.mean(chunk)))

    return pre_reward, post_reward, rand_reward, curve_windows


def main():
    print("=" * 65, flush=True)
    print("  NFSP on LEDUC-5 (389 info states)", flush=True)
    print(f"  Pretrain: {PRETRAIN}, Attack: {ATTACK_OUTER}x{ATTACK_INNER}, "
          f"Continued: {CONTINUED}", flush=True)
    print("=" * 65, flush=True)

    pre_all, post_all, rand_all = [], [], []

    for seed in SEEDS:
        print(f"\n  Seed {seed}:", flush=True)
        pre, post, rand, curve = run_one(seed)
        pre_all.append(pre)
        post_all.append(post)
        rand_all.append(rand)

    print(f"\n{'='*65}", flush=True)
    print(f"  NFSP LEDUC-5 RESULTS ({len(SEEDS)} seeds)", flush=True)
    print(f"{'='*65}", flush=True)

    def fmt(vals):
        m = np.mean(vals)
        ci = 1.96 * np.std(vals) / np.sqrt(len(vals))
        return f"{m:+.3f} +/- {ci:.3f}"

    print(f"  Pre-attack:    {fmt(pre_all)}", flush=True)
    print(f"  Post-advers:   {fmt(post_all)}", flush=True)
    print(f"  Post-random:   {fmt(rand_all)}", flush=True)
    delta_adv = np.mean(post_all) - np.mean(pre_all)
    delta_rand = np.mean(rand_all) - np.mean(pre_all)
    print(f"  Delta(adv):    {delta_adv:+.3f}", flush=True)
    print(f"  Delta(rand):   {delta_rand:+.3f}", flush=True)
    if delta_rand != 0:
        print(f"  Adv/Rand ratio: {delta_adv/delta_rand:.1f}x", flush=True)


if __name__ == "__main__":
    main()
