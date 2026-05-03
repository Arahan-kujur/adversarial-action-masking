"""Self-play training loop with action masking support."""

import numpy as np


def play_episode(env, agent, rng, cards=None):
    """Play one self-play episode. Returns (reward_p0, trajectory)."""
    env.reset(cards=cards, rng=rng)
    trajectory = []

    while not env.is_terminal:
        player = env.current_player
        info = env.info_state(player)
        legal = env.legal_actions()
        action = agent.select_action(info, legal, rng)
        trajectory.append((player, info, action))
        env.step(action)

    return env.returns[0], trajectory


def train_selfplay(env, agent, num_episodes, rng, mask_fn=None,
                   mask_after=None, callback=None):
    """Train agent via self-play, optionally applying a mask after a given episode.

    Parameters
    ----------
    env : MaskedKuhnPoker
    agent : QLearningAgent
    num_episodes : int
    rng : numpy Generator
    mask_fn : callable or None
        If provided, applied to env after `mask_after` episodes.
    mask_after : int or None
        Episode at which to activate the mask.
    callback : callable(episode, reward) or None
        Called after each episode.

    Returns
    -------
    rewards : list of float
    """
    rewards = []

    for ep in range(num_episodes):
        if mask_after is not None and ep == mask_after and mask_fn is not None:
            env.set_mask(mask_fn)

        reward, traj = play_episode(env, agent, rng)
        agent.update(traj, reward)
        rewards.append(reward)

        if callback:
            callback(ep, reward)

    return rewards
