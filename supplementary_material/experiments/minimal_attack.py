"""Minimal attack size experiment: how few states must the adversary target?

Sweeps adversary budget (number of info states it can mask at) from 0 to all,
comparing adversarial vs random masking at each budget level.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.envs.kuhn_poker import KuhnPokerEnv, MaskedKuhnPoker
from core.agents.q_learning import QLearningAgent
from core.training.selfplay import play_episode
from adversary.masking_policy import AdversarialMask, random_mask
from adversary.mask_utils import evaluate_agent


def run_budget_sweep(seeds=[42, 123, 456], pretrain=10000, adapt=10000,
                     eval_ep=5000, outer_iters=20):
    """Sweep adversary budget and compare against random baseline."""

    # Kuhn P0 has ~6 info states (card x history prefix)
    budgets = [0, 1, 2, 3, 6]

    print("=" * 60)
    print("  Minimal Attack Size: Adversarial vs Random")
    print("=" * 60)
    print(f"  {'Budget':>6s}  {'Adversarial':>12s}  {'Random':>12s}  {'Gap':>8s}")
    print("  " + "-" * 44)

    for budget in budgets:
        adv_rewards = []
        rand_rewards = []

        for seed in seeds:
            rng = np.random.default_rng(seed)

            # --- Adversarial ---
            env = MaskedKuhnPoker(KuhnPokerEnv())
            agent = QLearningAgent(num_actions=2, alpha=0.1, epsilon=0.15)
            adversary = AdversarialMask(target_player=0, num_actions=2,
                                        lr=0.01, budget=budget if budget > 0 else None)

            for _ in range(pretrain):
                r, t = play_episode(env, agent, rng)
                agent.update(t, r)

            if budget == 0:
                adv_r = evaluate_agent(env, agent, rng, eval_ep)
            else:
                env.set_mask(adversary.mask_fn)
                for outer in range(outer_iters):
                    batch = []
                    for _ in range(adapt // outer_iters):
                        r, t = play_episode(env, agent, rng)
                        agent.update(t, r)
                        batch.append((t, r))
                    adversary.update(batch)
                adv_r = evaluate_agent(env, agent, rng, eval_ep,
                                       mask_fn=adversary.mask_fn)
            adv_rewards.append(adv_r)

            # --- Random (same budget = same fraction of states masked) ---
            rng2 = np.random.default_rng(seed)
            env2 = MaskedKuhnPoker(KuhnPokerEnv())
            agent2 = QLearningAgent(num_actions=2, alpha=0.1, epsilon=0.15)

            for _ in range(pretrain):
                r, t = play_episode(env2, agent2, rng2)
                agent2.update(t, r)

            if budget == 0:
                rand_r = evaluate_agent(env2, agent2, rng2, eval_ep)
            else:
                prob = budget / 6.0
                rmask = random_mask(0, prob, np.random.default_rng(seed))
                env2.set_mask(rmask)
                for _ in range(adapt):
                    r, t = play_episode(env2, agent2, rng2)
                    agent2.update(t, r)
                rand_r = evaluate_agent(env2, agent2, rng2, eval_ep,
                                        mask_fn=rmask)
            rand_rewards.append(rand_r)

        adv_mean = np.mean(adv_rewards)
        rand_mean = np.mean(rand_rewards)
        gap = adv_mean - rand_mean

        print(f"  {budget:>6d}  {adv_mean:>+12.4f}  {rand_mean:>+12.4f}  {gap:>+8.4f}")

    # Also show the adversary's targeting for the full-budget case
    print("\n  Adversary targeting (full budget):")
    for info_state in sorted(adversary.theta.keys()):
        probs = adversary._get_removal_prob(info_state)
        removed = int(np.argmax(probs))
        name = "PASS" if removed == 0 else "BET"
        print(f"    {info_state:>4s}: remove {name} (p={probs[removed]:.3f})")


if __name__ == "__main__":
    run_budget_sweep()
