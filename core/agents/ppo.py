"""Tabular PPO agent with softmax policy over preference parameters."""

import numpy as np
from collections import defaultdict


class TabularPPOAgent:
    """Proximal Policy Optimization over a tabular softmax policy.

    Maintains a preference table ``theta[info_state][action]`` and derives
    action probabilities via softmax.  Uses the clipped surrogate objective
    with an entropy bonus, applied as a single gradient step per episode
    (on-policy).

    Parameters
    ----------
    num_actions : int
        Size of the action space.
    lr : float
        Learning rate for parameter updates.
    clip_eps : float
        PPO clipping parameter.
    entropy_coef : float
        Weight for the entropy bonus.
    """

    def __init__(
        self,
        num_actions: int = 2,
        lr: float = 0.01,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
    ):
        self.num_actions = num_actions
        self.lr = lr
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.theta: dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(self.num_actions)
        )

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _softmax(logits: np.ndarray, legal_actions: list) -> np.ndarray:
        """Masked softmax: only legal actions receive probability mass."""
        probs = np.zeros_like(logits)
        vals = np.array([logits[a] for a in legal_actions])
        vals = vals - vals.max()
        exp_vals = np.exp(vals)
        total = exp_vals.sum()
        for i, a in enumerate(legal_actions):
            probs[a] = exp_vals[i] / total
        return probs

    # -- public interface (same as QLearningAgent) -------------------------

    def select_action(
        self, info_state: str, legal_actions: list, rng: np.random.Generator
    ) -> int:
        """Sample an action from the current softmax policy."""
        probs = self._softmax(self.theta[info_state], legal_actions)
        p = np.array([probs[a] for a in legal_actions])
        return int(rng.choice(legal_actions, p=p))

    def policy_probs(self, info_state: str, legal_actions: list) -> np.ndarray:
        """Return the full action probability distribution."""
        return self._softmax(self.theta[info_state], legal_actions)

    def update(self, trajectory: list, reward_p0: float) -> None:
        """Run a single PPO gradient step on the completed episode.

        ``trajectory`` is a list of ``(player, info_state, action)`` tuples.
        Each tuple's reward is ``+reward_p0`` for player 0, ``-reward_p0``
        for player 1.  We treat the episode reward as the advantage
        (baseline-free, single-step return).

        For every (info_state, action) visited, the update:
          1. Computes the probability ratio  ``pi_new / pi_old``.
          2. Clips to ``[1 - eps, 1 + eps]``.
          3. Adds an entropy bonus.
          4. Takes a gradient ascent step on theta.
        """
        snapshots: dict[str, np.ndarray] = {}
        for _, info_state, _ in trajectory:
            if info_state not in snapshots:
                snapshots[info_state] = self.theta[info_state].copy()

        for player, info_state, action in trajectory:
            reward = reward_p0 if player == 0 else -reward_p0
            advantage = reward

            old_theta = snapshots[info_state]
            legal_actions = list(range(self.num_actions))

            old_probs = self._softmax(old_theta, legal_actions)
            old_prob = old_probs[action]
            if old_prob < 1e-8:
                continue

            cur_probs = self._softmax(self.theta[info_state], legal_actions)
            cur_prob = cur_probs[action]
            ratio = cur_prob / old_prob
            clipped_ratio = np.clip(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps)
            surrogate = min(ratio * advantage, clipped_ratio * advantage)

            entropy = -np.sum(
                cur_probs[a] * np.log(cur_probs[a] + 1e-10)
                for a in legal_actions
                if cur_probs[a] > 0
            )

            grad = np.zeros(self.num_actions)
            indicator = np.zeros(self.num_actions)
            indicator[action] = 1.0
            softmax_grad_log = indicator - cur_probs

            use_clipped = (clipped_ratio * advantage < ratio * advantage)
            if not use_clipped:
                grad += advantage * softmax_grad_log
            else:
                grad += 0.0  # clipped -- no gradient from surrogate

            entropy_grad = np.zeros(self.num_actions)
            for a in legal_actions:
                if cur_probs[a] > 0:
                    entropy_grad -= (np.log(cur_probs[a] + 1e-10) + 1.0) * (
                        (1.0 if a == action else 0.0) - cur_probs[a]
                    ) * cur_probs[action]

            entropy_grad_simple = np.zeros(self.num_actions)
            for a in legal_actions:
                if cur_probs[a] > 0:
                    d_entropy_a = 0.0
                    for b in legal_actions:
                        if cur_probs[b] > 0:
                            jacobian_ba = cur_probs[b] * ((1.0 if a == b else 0.0) - cur_probs[a])
                            d_entropy_a -= (np.log(cur_probs[b] + 1e-10) + 1.0) * jacobian_ba
                    entropy_grad_simple[a] = d_entropy_a

            grad += self.entropy_coef * entropy_grad_simple

            self.theta[info_state] += self.lr * grad
