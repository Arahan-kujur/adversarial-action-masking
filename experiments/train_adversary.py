"""Train adversarial masking policy against a self-play agent.

Bi-level optimisation:
  Inner loop: agent trains under adversary's mask
  Outer loop: adversary updates to minimise agent reward
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.envs.kuhn_poker import KuhnPokerEnv, MaskedKuhnPoker
from core.agents.q_learning import QLearningAgent
from core.training.selfplay import play_episode
from adversary.masking_policy import AdversarialMask
from adversary.mask_utils import evaluate_agent


def train_adversary(num_outer=20, inner_episodes=1000, eval_episodes=2000,
                    seed=42):
    rng = np.random.default_rng(seed)
    env = MaskedKuhnPoker(KuhnPokerEnv())
    agent = QLearningAgent(num_actions=2, alpha=0.1, epsilon=0.15)
    adversary = AdversarialMask(target_player=0, num_actions=2, lr=0.01)

    print("Pre-training agent (10k episodes, no mask)...")
    for _ in range(10000):
        reward, traj = play_episode(env, agent, rng)
        agent.update(traj, reward)

    baseline = evaluate_agent(env, agent, rng, eval_episodes)
    print(f"Baseline (no mask): {baseline:+.4f}")

    print(f"\nAdversarial training ({num_outer} outer iterations)...")
    env.set_mask(adversary.mask_fn)

    for outer in range(num_outer):
        batch = []
        for _ in range(inner_episodes):
            reward, traj = play_episode(env, agent, rng)
            agent.update(traj, reward)
            batch.append((traj, reward))

        adversary.update(batch)

        if (outer + 1) % 5 == 0:
            eval_r = evaluate_agent(env, agent, rng, eval_episodes,
                                    mask_fn=adversary.mask_fn)
            print(f"  Outer {outer+1:>3d}: agent reward = {eval_r:+.4f}")

    final = evaluate_agent(env, agent, rng, eval_episodes,
                           mask_fn=adversary.mask_fn)
    print(f"\nFinal under adversarial mask: {final:+.4f}")
    print(f"Baseline (no mask):          {baseline:+.4f}")
    print(f"Degradation:                 {final - baseline:+.4f}")

    print("\nAdversary's learned mask preferences:")
    for info_state in sorted(adversary.theta.keys()):
        probs = adversary._get_removal_prob(info_state)
        removed = int(np.argmax(probs))
        action_name = "PASS" if removed == 0 else "BET"
        print(f"  {info_state:>4s}: remove {action_name} (p={probs[removed]:.3f})")


def main():
    train_adversary()


if __name__ == "__main__":
    main()
