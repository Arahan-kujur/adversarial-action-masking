"""Learned perturbation adversary (fair RARL comparison).

Instead of a fixed p=0.3 perturbation, train a learned adversary
that decides WHEN to perturb and WHICH action to force, using the
same bi-level REINFORCE setup as the removal adversary.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from core.envs.kuhn_poker import KuhnPokerEnv, MaskedKuhnPoker
from core.agents.q_learning import QLearningAgent
from core.training.selfplay import play_episode
from adversary.masking_policy import AdversarialMask
from adversary.mask_utils import evaluate_agent

SEEDS = list(range(10))


class LearnedPerturbationAdversary:
    """Learned adversary that perturbs (overrides) actions rather than removing them.

    At each P0 state, decides:
      - which action to force (override victim's choice with this action)
      - or do nothing

    Uses softmax preference table, updated via REINFORCE.
    """
    def __init__(self, num_actions=2, lr=0.01):
        self.num_actions = num_actions
        self.lr = lr
        self.theta = {}  # info_state -> array of size num_actions+1 (force_0, force_1, ..., no_perturb)

    def _get_probs(self, info_state):
        if info_state not in self.theta:
            self.theta[info_state] = np.zeros(self.num_actions + 1)
        logits = self.theta[info_state]
        exp_l = np.exp(logits - logits.max())
        return exp_l / exp_l.sum()

    def perturb(self, info_state, chosen_action, legal_actions, player, rng):
        """Maybe override chosen_action. Returns actual action to execute."""
        if player != 0 or len(legal_actions) <= 1:
            return chosen_action
        probs = self._get_probs(info_state)
        idx = int(rng.choice(len(probs), p=probs))
        if idx < self.num_actions and idx in legal_actions:
            return idx  # force this action
        return chosen_action  # no perturbation

    def update(self, trajectories_and_rewards):
        for traj, reward in trajectories_and_rewards:
            for player, info_state, action in traj:
                if player != 0:
                    continue
                probs = self._get_probs(info_state)
                grad = -probs.copy()
                chosen = np.argmax(probs)
                grad[chosen] += 1.0
                self.theta[info_state] += self.lr * (-reward) * grad


def play_episode_perturbed(env, agent, adversary, rng):
    """Play episode with perturbation adversary overriding actions."""
    env.reset(rng=rng)
    trajectory = []
    while not env.is_terminal:
        player = env.current_player
        info = env.info_state(player)
        legal = env.legal_actions()
        action = agent.select_action(info, legal, rng)
        actual = adversary.perturb(info, action, legal, player, rng)
        trajectory.append((player, info, actual))
        env.step(actual)
    return env.returns[0], trajectory


def run_one(mode, seed):
    rng = np.random.default_rng(seed)
    env = MaskedKuhnPoker(KuhnPokerEnv())
    agent = QLearningAgent(num_actions=2, alpha=0.1, epsilon=0.15)

    for _ in range(10000):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)

    if mode == "none":
        return evaluate_agent(env, agent, rng, 5000)

    if mode == "learned_perturb":
        adv = LearnedPerturbationAdversary(num_actions=2, lr=0.01)
        for _ in range(20):
            batch = []
            for _ in range(500):
                r, t = play_episode_perturbed(env, agent, adv, rng)
                agent.update(t, r)
                batch.append((t, r))
            adv.update(batch)
        rewards = []
        for _ in range(5000):
            r, t = play_episode_perturbed(env, agent, adv, rng)
            rewards.append(r)
        return np.mean(rewards)

    if mode == "learned_removal":
        adv = AdversarialMask(target_player=0, num_actions=2, lr=0.01)
        env.set_mask(adv.mask_fn)
        for _ in range(20):
            batch = []
            for _ in range(500):
                r, t = play_episode(env, agent, rng)
                agent.update(t, r)
                batch.append((t, r))
            adv.update(batch)
        return evaluate_agent(env, agent, rng, 5000, mask_fn=adv.mask_fn)


def main():
    print("Learned Perturbation vs Learned Removal (10 seeds)", flush=True)
    print(f"{'Mode':>20s}  {'Mean':>8s}  {'CI':>8s}", flush=True)
    print("-" * 45, flush=True)
    for mode in ["none", "learned_perturb", "learned_removal"]:
        vals = [run_one(mode, s) for s in SEEDS]
        m = np.mean(vals)
        ci = 1.96 * np.std(vals) / np.sqrt(len(vals))
        print(f"{mode:>20s}  {m:+.4f}  +/-{ci:.4f}", flush=True)


if __name__ == "__main__":
    main()
