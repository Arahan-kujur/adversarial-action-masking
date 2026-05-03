"""Utilities for action mask analysis and evaluation."""

import numpy as np


def compute_mask_rate(mask_fn, env, agent, rng, num_episodes=1000):
    """Measure how often the mask actually removes actions.

    Returns dict with:
        mask_rate: fraction of P0 decisions where an action was removed
        avg_actions_removed: mean number of actions removed per decision
    """
    total_decisions = 0
    masked_decisions = 0
    total_removed = 0

    for _ in range(num_episodes):
        env.reset(rng=rng)
        while not env.is_terminal:
            player = env.current_player
            info = env.info_state(player)
            base_legal = env.env.legal_actions()
            masked_legal = mask_fn(info, base_legal, player)

            if player == 0:
                total_decisions += 1
                removed = len(base_legal) - len(masked_legal)
                if removed > 0:
                    masked_decisions += 1
                    total_removed += removed

            action = agent.select_action(info, masked_legal, rng)
            env.step(action)

    return {
        "mask_rate": masked_decisions / max(total_decisions, 1),
        "avg_actions_removed": total_removed / max(total_decisions, 1),
    }


def evaluate_agent(env, agent, rng, num_episodes=2000, mask_fn=None):
    """Evaluate agent reward under optional masking. Returns mean reward."""
    if mask_fn:
        env.set_mask(mask_fn)
    else:
        env.set_mask(None)

    rewards = []
    for _ in range(num_episodes):
        env.reset(rng=rng)
        traj = []
        while not env.is_terminal:
            player = env.current_player
            info = env.info_state(player)
            legal = env.legal_actions()
            action = agent.select_action(info, legal, rng)
            traj.append((player, info, action))
            env.step(action)
        rewards.append(env.returns[0])

    env.set_mask(None)
    return float(np.mean(rewards))
