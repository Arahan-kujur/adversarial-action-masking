"""Budget fairness analysis: compute effective L0 support for each masking strategy.

For the neural adversary, random(p=0.5), and fixed removal in Leduc,
report exactly how many info states are actually masked.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.envs.leduc_poker import LeducPokerEnv, MaskedLeducPoker, NUM_ACTIONS
from core.agents.dqn import DQNAgent, get_encoder
from core.agents.q_learning import QLearningAgent
from core.training.selfplay import play_episode
from adversary.masking_policy import AdversarialMask, random_mask, fixed_removal
from adversary.mask_utils import evaluate_agent


def measure_effective_budget(env, agent, mask_fn, rng, num_episodes=5000):
    """Count distinct info states where the mask actually removes an action."""
    masked_states = set()
    all_p0_states = set()

    for _ in range(num_episodes):
        env.env.reset(rng=rng)
        while not env.env.is_terminal:
            player = env.env.current_player
            info = env.env.info_state(player)
            base_legal = env.env.legal_actions()

            if player == 0:
                all_p0_states.add(info)
                masked_legal = mask_fn(info, base_legal, player)
                if len(masked_legal) < len(base_legal):
                    masked_states.add(info)

            legal = env.legal_actions()
            action = agent.select_action(info, legal, rng)
            env.env.step(action)

    return len(masked_states), len(all_p0_states)


def main():
    print("=" * 65, flush=True)
    print("Budget Fairness Analysis: Effective L0 Support", flush=True)
    print("=" * 65, flush=True)

    seeds = list(range(5))

    # --- Kuhn ---
    from core.envs.kuhn_poker import KuhnPokerEnv, MaskedKuhnPoker

    print("\n--- KUHN POKER ---", flush=True)
    for seed in seeds:
        rng = np.random.default_rng(seed)
        env = MaskedKuhnPoker(KuhnPokerEnv())
        agent = QLearningAgent(num_actions=2, alpha=0.1, epsilon=0.15)
        for _ in range(10000):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)

        # Adversarial (full budget)
        adv = AdversarialMask(target_player=0, num_actions=2, lr=0.01)
        env.set_mask(adv.mask_fn)
        for _ in range(20):
            batch = []
            for _ in range(500):
                r, t = play_episode(env, agent, rng)
                agent.update(t, r)
                batch.append((t, r))
            adv.update(batch)
        k_adv, total = measure_effective_budget(env, agent, adv.mask_fn, rng)

        # Random
        rm = random_mask(0, 0.5, np.random.default_rng(seed + 99))
        samples = []
        for _ in range(20):
            k_r, _ = measure_effective_budget(env, agent, rm, rng, 1000)
            samples.append(k_r)
        k_rand = np.mean(samples)

        # Fixed
        fm = fixed_removal(0, 1)
        k_fixed, _ = measure_effective_budget(env, agent, fm, rng)

        print(f"  seed={seed}: total_P0_states={total}, "
              f"adv_L0={k_adv}, random_L0={k_rand:.1f}, fixed_L0={k_fixed}",
              flush=True)

    # --- Leduc ---
    print("\n--- LEDUC POKER ---", flush=True)
    for seed in seeds[:3]:
        rng = np.random.default_rng(seed)
        env = MaskedLeducPoker(LeducPokerEnv())
        agent = QLearningAgent(num_actions=NUM_ACTIONS, alpha=0.1, epsilon=0.15)
        for _ in range(15000):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)

        # Tabular adversarial (full budget)
        adv = AdversarialMask(target_player=0, num_actions=NUM_ACTIONS, lr=0.01)
        env.set_mask(adv.mask_fn)
        for _ in range(20):
            batch = []
            for _ in range(500):
                r, t = play_episode(env, agent, rng)
                agent.update(t, r)
                batch.append((t, r))
            adv.update(batch)
        k_adv, total = measure_effective_budget(env, agent, adv.mask_fn, rng)

        # Random
        rm = random_mask(0, 0.5, np.random.default_rng(seed + 99))
        samples = []
        for _ in range(20):
            k_r, _ = measure_effective_budget(env, agent, rm, rng, 1000)
            samples.append(k_r)
        k_rand = np.mean(samples)

        # Fixed (remove RAISE)
        fm = fixed_removal(0, 2)
        k_fixed, _ = measure_effective_budget(env, agent, fm, rng)

        print(f"  seed={seed}: total_P0_states={total}, "
              f"adv_L0={k_adv}, random_L0={k_rand:.1f}, fixed_L0={k_fixed}",
              flush=True)


if __name__ == "__main__":
    main()
