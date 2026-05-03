"""Attack generalization: train adversary on one seed, test on others.

Tests whether the learned mask transfers to unseen agent policies.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.envs.kuhn_poker import KuhnPokerEnv, MaskedKuhnPoker
from core.agents.q_learning import QLearningAgent
from core.training.selfplay import play_episode
from adversary.masking_policy import AdversarialMask
from adversary.mask_utils import evaluate_agent


def main():
    train_seed = 42
    test_seeds = [123, 456, 789, 1024, 2048]

    # Train adversary on one agent
    rng = np.random.default_rng(train_seed)
    env = MaskedKuhnPoker(KuhnPokerEnv())
    agent = QLearningAgent(num_actions=2, alpha=0.1, epsilon=0.15)

    for _ in range(10000):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)

    adversary = AdversarialMask(target_player=0, num_actions=2, lr=0.01)
    env.set_mask(adversary.mask_fn)

    for outer in range(20):
        batch = []
        for _ in range(500):
            r, t = play_episode(env, agent, rng)
            agent.update(t, r)
            batch.append((t, r))
        adversary.update(batch)

    train_r = evaluate_agent(env, agent, rng, 5000, mask_fn=adversary.mask_fn)

    print("=" * 60)
    print("  Attack Generalization: Train Once, Test on New Agents")
    print("=" * 60)
    print(f"  Train seed {train_seed}: {train_r:+.4f}")
    print()

    # Test on fresh agents (different seeds, different Q-tables)
    transfer_rewards = []
    retrain_rewards = []

    for test_seed in test_seeds:
        rng_test = np.random.default_rng(test_seed)
        env_test = MaskedKuhnPoker(KuhnPokerEnv())
        agent_test = QLearningAgent(num_actions=2, alpha=0.1, epsilon=0.15)

        for _ in range(10000):
            r, t = play_episode(env_test, agent_test, rng_test)
            agent_test.update(t, r)

        # Transfer: use the SAME adversary mask (trained on seed 42)
        env_test.set_mask(adversary.mask_fn)
        for _ in range(10000):
            r, t = play_episode(env_test, agent_test, rng_test)
            agent_test.update(t, r)
        tr = evaluate_agent(env_test, agent_test, rng_test, 5000,
                            mask_fn=adversary.mask_fn)
        transfer_rewards.append(tr)

        # Retrain: train a NEW adversary for this agent
        rng_rt = np.random.default_rng(test_seed + 10000)
        env_rt = MaskedKuhnPoker(KuhnPokerEnv())
        agent_rt = QLearningAgent(num_actions=2, alpha=0.1, epsilon=0.15)
        for _ in range(10000):
            r, t = play_episode(env_rt, agent_rt, rng_rt)
            agent_rt.update(t, r)

        adv_new = AdversarialMask(target_player=0, num_actions=2, lr=0.01)
        env_rt.set_mask(adv_new.mask_fn)
        for outer in range(20):
            batch = []
            for _ in range(500):
                r, t = play_episode(env_rt, agent_rt, rng_rt)
                agent_rt.update(t, r)
                batch.append((t, r))
            adv_new.update(batch)
        rr = evaluate_agent(env_rt, agent_rt, rng_rt, 5000,
                            mask_fn=adv_new.mask_fn)
        retrain_rewards.append(rr)

        print(f"  Seed {test_seed}: transfer={tr:+.4f}  retrained={rr:+.4f}")

    print(f"\n  Transfer mean:   {np.mean(transfer_rewards):+.4f} +/- {np.std(transfer_rewards):.4f}")
    print(f"  Retrained mean:  {np.mean(retrain_rewards):+.4f} +/- {np.std(retrain_rewards):.4f}")
    print(f"  Gap:             {np.mean(transfer_rewards) - np.mean(retrain_rewards):+.4f}")
    print("\n  Small gap = adversary learned structural patterns, not agent-specific exploits.")


if __name__ == "__main__":
    main()
