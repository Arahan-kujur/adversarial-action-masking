"""Compare masking strategies: none, random, fixed, adversarial."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.envs.kuhn_poker import KuhnPokerEnv, MaskedKuhnPoker
from core.agents.q_learning import QLearningAgent
from core.training.selfplay import play_episode
from adversary.masking_policy import (
    no_mask, random_mask, fixed_removal, AdversarialMask,
)
from adversary.mask_utils import evaluate_agent


def run_comparison(seed=42, pretrain=10000, adapt=10000, eval_ep=5000):
    rng = np.random.default_rng(seed)

    results = {}

    for label, mask_factory in [
        ("No mask", lambda: None),
        ("Random (p=0.3)", lambda: random_mask(0, 0.3, np.random.default_rng(seed))),
        ("Random (p=0.7)", lambda: random_mask(0, 0.7, np.random.default_rng(seed))),
        ("Fixed (remove BET)", lambda: fixed_removal(0, 1)),
        ("Adversarial", "adversarial"),
    ]:
        rng_local = np.random.default_rng(seed)
        env = MaskedKuhnPoker(KuhnPokerEnv())
        agent = QLearningAgent(num_actions=2, alpha=0.1, epsilon=0.15)

        for _ in range(pretrain):
            reward, traj = play_episode(env, agent, rng_local)
            agent.update(traj, reward)

        if label == "Adversarial":
            adversary = AdversarialMask(target_player=0, num_actions=2, lr=0.01)
            env.set_mask(adversary.mask_fn)
            for outer in range(20):
                batch = []
                for _ in range(adapt // 20):
                    reward, traj = play_episode(env, agent, rng_local)
                    agent.update(traj, reward)
                    batch.append((traj, reward))
                adversary.update(batch)
            mask_fn = adversary.mask_fn
        else:
            mask_fn = mask_factory()
            if mask_fn:
                env.set_mask(mask_fn)
                for _ in range(adapt):
                    reward, traj = play_episode(env, agent, rng_local)
                    agent.update(traj, reward)

        r = evaluate_agent(env, agent, rng_local, eval_ep, mask_fn=mask_fn)
        results[label] = r

    print("=" * 50)
    print("  Masking Strategy Comparison")
    print("=" * 50)
    print(f"  {'Strategy':<25s}  {'P0 Reward':>10s}")
    print("  " + "-" * 40)
    for label, r in results.items():
        print(f"  {label:<25s}  {r:>+10.4f}")


def main():
    run_comparison()


if __name__ == "__main__":
    main()
