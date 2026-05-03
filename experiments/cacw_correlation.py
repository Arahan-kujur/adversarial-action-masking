"""CACw vs reward correlation: empirical proof that adversary minimizes CACw.

For each budget k=0..6 in Kuhn, compute:
  - CACw (reach-weighted fraction of states with >1 legal action)
  - victim reward under adversarial mask at that budget

Then print correlation and data for plotting.
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


def compute_cacw(mask_fn, env):
    """Compute reach-weighted CAC under a mask."""
    cacw = 0.0
    for state in P0_STATES:
        legal = [0, 1]  # both always legal in base game
        masked = mask_fn(state, legal, 0)
        if len(masked) > 1:
            cacw += REACH_WEIGHTS.get(state, 1/6)
    return cacw


def run_budget(budget, mode, seed):
    rng = np.random.default_rng(seed)
    env = MaskedKuhnPoker(KuhnPokerEnv())
    agent = QLearningAgent(num_actions=2, alpha=0.1, epsilon=0.15)

    for _ in range(10000):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)

    if budget == 0:
        reward = evaluate_agent(env, agent, rng, 5000)
        cacw = sum(REACH_WEIGHTS.values())  # all states have >1 action
        return cacw, reward

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
        cacw = compute_cacw(adv.mask_fn, env)
        return cacw, reward

    elif mode == "random":
        rm = random_mask(0, budget / 6.0, np.random.default_rng(seed + 99))
        env.set_mask(rm)
        for _ in range(15000):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)
        reward = evaluate_agent(env, agent, rng, 5000, mask_fn=rm)
        cacw_samples = []
        for _ in range(1000):
            c = 0.0
            for state in P0_STATES:
                masked = rm(state, [0, 1], 0)
                if len(masked) > 1:
                    c += REACH_WEIGHTS.get(state, 1/6)
            cacw_samples.append(c)
        cacw = np.mean(cacw_samples)
        return cacw, reward


def main():
    print("CACw vs Reward Correlation (Kuhn, 10 seeds)", flush=True)
    print(f"{'Budget':>6s}  {'Mode':>12s}  {'CACw':>8s}  {'Reward':>10s}", flush=True)
    print("-" * 50, flush=True)

    adv_cacw, adv_reward = [], []
    rand_cacw, rand_reward = [], []

    for budget in range(7):
        for mode in ["adversarial", "random"]:
            cacw_vals, reward_vals = [], []
            for seed in SEEDS:
                c, r = run_budget(budget, mode, seed)
                cacw_vals.append(c)
                reward_vals.append(r)

            mc = np.mean(cacw_vals)
            mr = np.mean(reward_vals)
            print(f"{budget:>6d}  {mode:>12s}  {mc:>8.4f}  {mr:>+10.4f}", flush=True)

            if mode == "adversarial":
                adv_cacw.extend(cacw_vals)
                adv_reward.extend(reward_vals)
            else:
                rand_cacw.extend(cacw_vals)
                rand_reward.extend(reward_vals)

    # Compute correlations
    all_cacw = adv_cacw + rand_cacw
    all_reward = adv_reward + rand_reward

    corr = np.corrcoef(all_cacw, all_reward)[0, 1]
    adv_corr = np.corrcoef(adv_cacw, adv_reward)[0, 1]

    print(f"\nOverall Pearson r(CACw, reward) = {corr:.4f}", flush=True)
    print(f"Adversarial-only Pearson r      = {adv_corr:.4f}", flush=True)
    print(f"\nInterpretation: positive r means higher CACw -> higher reward", flush=True)
    print(f"(adversary minimizes CACw -> minimizes reward)", flush=True)


if __name__ == "__main__":
    main()
