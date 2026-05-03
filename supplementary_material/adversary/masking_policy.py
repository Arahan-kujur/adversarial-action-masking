"""Action masking policies: random, fixed, and adversarial."""

import numpy as np


def no_mask(info_state, legal_actions, player):
    """No masking -- return all legal actions."""
    return legal_actions


def random_mask(target_player=0, remove_prob=0.5, rng=None):
    """Randomly remove actions with probability `remove_prob` for `target_player`.

    Returns a mask function compatible with MaskedKuhnPoker.set_mask().
    """
    _rng = rng or np.random.default_rng()

    def mask_fn(info_state, legal_actions, player):
        if player != target_player or len(legal_actions) <= 1:
            return legal_actions
        kept = [a for a in legal_actions if _rng.random() > remove_prob]
        return kept if kept else [legal_actions[0]]

    return mask_fn


def fixed_removal(target_player=0, removed_action=1):
    """Always remove a specific action from the target player.

    Default: remove BET (action 1) from player 0.
    """
    def mask_fn(info_state, legal_actions, player):
        if player != target_player:
            return legal_actions
        filtered = [a for a in legal_actions if a != removed_action]
        return filtered if filtered else legal_actions

    return mask_fn


class AdversarialMask:
    """Learned adversarial masking policy.

    The adversary observes the agent's info state and chooses which
    action to remove. Trained to minimise the agent's reward.

    Parameters
    ----------
    target_player : int
    num_actions : int
    lr : float
    budget : int or None
        Max number of info states the adversary can mask at.
        If None, no budget limit (can mask at every state).
    """

    def __init__(self, target_player=0, num_actions=2, lr=0.01, budget=None):
        self.target_player = target_player
        self.num_actions = num_actions
        self.lr = lr
        self.budget = budget
        self.theta = {}

    def _get_removal_prob(self, info_state):
        """Probability of removing each action (softmax over theta)."""
        if info_state not in self.theta:
            self.theta[info_state] = np.zeros(self.num_actions)
        logits = self.theta[info_state]
        exp_l = np.exp(logits - logits.max())
        return exp_l / exp_l.sum()

    def _active_states(self):
        """Return the top-budget states by adversary confidence."""
        if self.budget is None or not self.theta:
            return None
        confidences = {}
        for info_state, th in self.theta.items():
            probs = np.exp(th - th.max())
            probs /= probs.sum()
            confidences[info_state] = max(probs) - 1.0 / self.num_actions
        ranked = sorted(confidences, key=confidences.get, reverse=True)
        return set(ranked[:self.budget])

    def mask_fn(self, info_state, legal_actions, player):
        """Mask function compatible with MaskedEnv.set_mask()."""
        if player != self.target_player or len(legal_actions) <= 1:
            return legal_actions

        if self.budget is not None:
            active = self._active_states()
            if active is not None and info_state not in active:
                return legal_actions

        probs = self._get_removal_prob(info_state)
        remove_idx = np.argmax(probs)

        if remove_idx in legal_actions:
            filtered = [a for a in legal_actions if a != remove_idx]
            if filtered:
                return filtered
        return legal_actions

    def update(self, trajectories_and_rewards):
        """Update adversary to minimise agent reward.

        Parameters
        ----------
        trajectories_and_rewards : list of (trajectory, reward_p0)
            Each trajectory is [(player, info_state, action), ...].
        """
        for traj, reward in trajectories_and_rewards:
            for player, info_state, action in traj:
                if player != self.target_player:
                    continue
                probs = self._get_removal_prob(info_state)
                grad = -probs.copy()
                removed = np.argmax(probs)
                grad[removed] += 1.0
                self.theta[info_state] += self.lr * (-reward) * grad
