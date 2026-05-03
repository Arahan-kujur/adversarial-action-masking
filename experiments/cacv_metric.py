"""Counterfactual-Value-weighted CAC (CACv) vs plain CACw.

CACv weights each info state by its counterfactual value sensitivity:
  CACv = sum_h rho(h) * |CFV(h)| * 1[|M(h)| > 1]

where CFV(h) is the counterfactual value difference between actions.
This should correlate better with reward than plain CACw because it
captures WHICH states matter for value, not just reachability.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.envs.kuhn_poker import KuhnPokerEnv, MaskedKuhnPoker
from core.agents.q_learning import QLearningAgent
from core.training.selfplay import play_episode
from adversary.masking_policy import AdversarialMask, random_mask
from adversary.mask_utils import evaluate_agent

P0_STATES = ["0", "0pb", "1", "1pb", "2", "2pb"]
REACH_WEIGHTS = {
    "0": 1/3, "0pb": 1/3 * 0.5,
    "1": 1/3, "1pb": 1/3 * 0.5,
    "2": 1/3, "2pb": 1/3 * 0.5,
}
SEEDS = list(range(10))


def compute_cacw(mask_fn):
    cacw = 0.0
    for state in P0_STATES:
        masked = mask_fn(state, [0, 1], 0)
        if len(masked) > 1:
            cacw += REACH_WEIGHTS.get(state, 1/6)
    return cacw


def compute_cacv(mask_fn, agent):
    """CACv: reach * |Q-value gap| * indicator(>1 action available)."""
    cacv = 0.0
    for state in P0_STATES:
        masked = mask_fn(state, [0, 1], 0)
        if len(masked) > 1:
            q = agent.q[state]
            value_gap = abs(q[0] - q[1])
            cacv += REACH_WEIGHTS.get(state, 1/6) * value_gap
    return cacv


def run_budget(budget, mode, seed):
    rng = np.random.default_rng(seed)
    env = MaskedKuhnPoker(KuhnPokerEnv())
    agent = QLearningAgent(num_actions=2, alpha=0.1, epsilon=0.15)

    for _ in range(10000):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)

    if budget == 0:
        reward = evaluate_agent(env, agent, rng, 5000)
        identity = lambda s, a, p: a
        cacw = compute_cacw(identity)
        cacv = compute_cacv(identity, agent)
        return cacw, cacv, reward

    if mode == "adversarial":
        adv = AdversarialMask(target_player=0, num_actions=2, lr=0.01,
                              budget=budget)
        env.set_mask(adv.mask_fn)
        for _ in range(30):
            batch = []
            for _ in range(500):
                r, t = play_episode(env, agent, rng)
                agent.update(t, r)
                batch.append((t, r))
            adv.update(batch)
        reward = evaluate_agent(env, agent, rng, 5000, mask_fn=adv.mask_fn)
        cacw = compute_cacw(adv.mask_fn)
        cacv = compute_cacv(adv.mask_fn, agent)
        return cacw, cacv, reward

    elif mode == "random":
        rm = random_mask(0, budget / 6.0, np.random.default_rng(seed + 99))
        env.set_mask(rm)
        for _ in range(15000):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)
        reward = evaluate_agent(env, agent, rng, 5000, mask_fn=rm)
        cacw_samples, cacv_samples = [], []
        for _ in range(500):
            cacw_samples.append(compute_cacw(rm))
            cacv_samples.append(compute_cacv(rm, agent))
        return np.mean(cacw_samples), np.mean(cacv_samples), reward


def main():
    print("CACw vs CACv Correlation (Kuhn, 10 seeds)", flush=True)
    print(f"{'Budget':>6s}  {'Mode':>12s}  {'CACw':>8s}  {'CACv':>8s}  {'Reward':>10s}",
          flush=True)
    print("-" * 55, flush=True)

    all_cacw, all_cacv, all_reward = [], [], []

    for budget in range(7):
        for mode in ["adversarial", "random"]:
            cw_vals, cv_vals, r_vals = [], [], []
            for seed in SEEDS:
                cw, cv, r = run_budget(budget, mode, seed)
                cw_vals.append(cw)
                cv_vals.append(cv)
                r_vals.append(r)

            mcw = np.mean(cw_vals)
            mcv = np.mean(cv_vals)
            mr = np.mean(r_vals)
            print(f"{budget:>6d}  {mode:>12s}  {mcw:>8.4f}  {mcv:>8.4f}  {mr:>+10.4f}",
                  flush=True)

            all_cacw.extend(cw_vals)
            all_cacv.extend(cv_vals)
            all_reward.extend(r_vals)

    r_cacw = np.corrcoef(all_cacw, all_reward)[0, 1]
    r_cacv = np.corrcoef(all_cacv, all_reward)[0, 1]

    print(f"\nPearson r(CACw, reward) = {r_cacw:.4f}", flush=True)
    print(f"Pearson r(CACv, reward) = {r_cacv:.4f}", flush=True)
    print(f"\nCACv improvement: {r_cacv - r_cacw:+.4f}", flush=True)


if __name__ == "__main__":
    main()
