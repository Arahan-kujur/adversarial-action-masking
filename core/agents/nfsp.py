"""Tabular Neural Fictitious Self-Play (NFSP) agent.

Dual-policy agent: best-response (Q-learning) + average strategy (action counts).
"""
import numpy as np
from collections import defaultdict


class NFSPAgent:
    def __init__(self, num_actions=2, alpha=0.1, epsilon=0.15, eta=0.1):
        self.num_actions = num_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.eta = eta  # prob of playing best-response vs average
        self.q = defaultdict(lambda: np.zeros(num_actions))
        self.avg_counts = defaultdict(lambda: np.zeros(num_actions))

    def select_action(self, info_state, legal_actions, rng):
        if rng.random() < self.eta:
            return self._best_response_action(info_state, legal_actions, rng)
        else:
            return self._average_action(info_state, legal_actions, rng)

    def _best_response_action(self, info_state, legal_actions, rng):
        if rng.random() < self.epsilon:
            return int(rng.choice(legal_actions))
        q_vals = self.q[info_state]
        legal_q = [(a, q_vals[a]) for a in legal_actions]
        best_val = max(v for _, v in legal_q)
        best_actions = [a for a, v in legal_q if abs(v - best_val) < 1e-8]
        return int(rng.choice(best_actions))

    def _average_action(self, info_state, legal_actions, rng):
        counts = self.avg_counts[info_state][legal_actions]
        total = counts.sum()
        if total < 1:
            return int(rng.choice(legal_actions))
        probs = counts / total
        return int(rng.choice(legal_actions, p=probs))

    def update(self, trajectory, reward_p0):
        for player, info_state, action in trajectory:
            r = reward_p0 if player == 0 else -reward_p0
            self.q[info_state][action] += self.alpha * (r - self.q[info_state][action])
            self.avg_counts[info_state][action] += 1.0

    def policy_probs(self, info_state, legal_actions):
        counts = self.avg_counts[info_state][legal_actions]
        total = counts.sum()
        if total < 1:
            return {a: 1.0 / len(legal_actions) for a in legal_actions}
        probs = counts / total
        return {a: float(probs[i]) for i, a in enumerate(legal_actions)}
