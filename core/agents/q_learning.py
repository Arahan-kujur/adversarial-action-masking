"""Tabular Q-learning agent."""

import numpy as np
from collections import defaultdict


class QLearningAgent:
    """Epsilon-greedy tabular Q-learning with MC-style terminal updates."""

    def __init__(
        self,
        num_actions=2,
        alpha=0.1,
        epsilon=0.15,
        regularize=False,
        high_cacv_states=None,
        reg_tau=0.5,
        reg_lambda=0.01,
    ):
        self.num_actions = num_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.q = defaultdict(lambda: np.zeros(self.num_actions))
        self.regularize = regularize
        self.high_cacv_states = set(high_cacv_states or [])
        self.reg_tau = reg_tau
        self.reg_lambda = reg_lambda

    def set_regularization_states(self, states):
        self.high_cacv_states = set(states)

    def select_action(self, info_state, legal_actions, rng):
        if rng.random() < self.epsilon:
            return int(rng.choice(legal_actions))
        q_vals = self.q[info_state].copy()
        masked = np.full(self.num_actions, -np.inf)
        for a in legal_actions:
            masked[a] = q_vals[a]
        return int(np.argmax(masked))

    def update(self, trajectory, reward_p0):
        for player, info_state, action in trajectory:
            r = reward_p0 if player == 0 else -reward_p0
            self.q[info_state][action] += self.alpha * (r - self.q[info_state][action])
            if self.regularize and info_state in self.high_cacv_states:
                q_vals = self.q[info_state]
                delta = float(q_vals.max() - q_vals.min())
                if delta > self.reg_tau:
                    self.q[info_state] += self.reg_lambda * (q_vals.mean() - q_vals)

    def policy_probs(self, info_state, legal_actions):
        """Return action probabilities at this state (for adversary training)."""
        probs = np.zeros(self.num_actions)
        q_vals = self.q[info_state]
        best = max(legal_actions, key=lambda a: q_vals[a])
        for a in legal_actions:
            probs[a] = self.epsilon / len(legal_actions)
        probs[best] += 1.0 - self.epsilon
        return probs
