"""Train agent via self-play, then evaluate under masking."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.envs.kuhn_poker import KuhnPokerEnv, MaskedKuhnPoker
from core.agents.q_learning import QLearningAgent
from core.training.selfplay import train_selfplay
from adversary.masking_policy import no_mask, fixed_removal
from adversary.mask_utils import evaluate_agent


def main():
    seed = 42
    rng = np.random.default_rng(seed)

    env = MaskedKuhnPoker(KuhnPokerEnv())
    agent = QLearningAgent(num_actions=2, alpha=0.1, epsilon=0.15)

    print("Training self-play (20k episodes)...")
    rewards = train_selfplay(env, agent, 20000, rng)

    pre_reward = evaluate_agent(env, agent, rng, num_episodes=5000)
    print(f"Pre-perturbation reward: {pre_reward:+.4f}")

    print("\nApplying fixed mask (remove BET from P0)...")
    mask = fixed_removal(target_player=0, removed_action=1)
    post_reward = evaluate_agent(env, agent, rng, num_episodes=5000, mask_fn=mask)
    print(f"Post-mask reward:        {post_reward:+.4f}")

    print("\nContinuing training under mask (10k episodes)...")
    rewards_masked = train_selfplay(env, agent, 10000, rng, mask_fn=mask, mask_after=0)
    adapted_reward = evaluate_agent(env, agent, rng, num_episodes=5000, mask_fn=mask)
    print(f"Adapted reward:          {adapted_reward:+.4f}")


if __name__ == "__main__":
    main()
