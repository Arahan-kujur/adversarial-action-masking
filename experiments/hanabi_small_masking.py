"""Hanabi-Small action removal benchmark."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adversary.masking_policy import AdversarialMask, random_mask
from core.agents.q_learning import QLearningAgent
from core.envs.hanabi_small import HanabiSmallEnv, MaskedHanabiSmall, NUM_ACTIONS, HINT_COLOR, HINT_RANK


SEEDS = list(range(5))
PRETRAIN = 10000
ATTACK_OUTER = 12
ATTACK_INNER = 400
EVAL = 3000


class LearnedPerturbation:
    def __init__(self, num_actions=NUM_ACTIONS, lr=0.01):
        self.num_actions = num_actions
        self.lr = lr
        self.theta = {}

    def _probs(self, info_state):
        if info_state not in self.theta:
            self.theta[info_state] = np.zeros(self.num_actions + 1)
        logits = self.theta[info_state]
        exp = np.exp(logits - logits.max())
        return exp / exp.sum()

    def perturb(self, info_state, action, legal_actions, player, rng):
        if player != 0 or len(legal_actions) <= 1:
            return action
        probs = self._probs(info_state)
        idx = int(rng.choice(len(probs), p=probs))
        if idx < self.num_actions and idx in legal_actions:
            return idx
        return action

    def update(self, batch):
        for traj, reward in batch:
            for player, info, _action in traj:
                if player != 0:
                    continue
                probs = self._probs(info)
                grad = -probs.copy()
                chosen = int(np.argmax(probs))
                grad[chosen] += 1.0
                self.theta[info] += self.lr * (-reward) * grad


def play_episode(env, agent, rng, perturb=None):
    env.reset(rng=rng)
    traj = []
    while not env.is_terminal:
        p = env.current_player
        info = env.info_state(p)
        legal = env.legal_actions()
        action = agent.select_action(info, legal, rng)
        if perturb is not None:
            action = perturb.perturb(info, action, legal, p, rng)
        traj.append((p, info, action))
        env.step(action)
    return env.returns[0], traj, env.metrics


def coop_update(agent, traj, reward):
    for _player, info, action in traj:
        agent.q[info][action] += agent.alpha * (reward - agent.q[info][action])


def train_base(seed):
    rng = np.random.default_rng(seed)
    env = MaskedHanabiSmall(HanabiSmallEnv())
    agent = QLearningAgent(num_actions=NUM_ACTIONS, alpha=0.08, epsilon=0.15)
    for _ in range(PRETRAIN):
        r, t, _ = play_episode(env, agent, rng)
        coop_update(agent, t, r)
    return env, agent, rng


def learn_removal(seed, agent):
    rng = np.random.default_rng(seed + 1000)
    env = MaskedHanabiSmall(HanabiSmallEnv())
    adv = AdversarialMask(target_player=0, num_actions=NUM_ACTIONS, lr=0.01)
    env.set_mask(adv.mask_fn)
    for _ in range(ATTACK_OUTER):
        batch = []
        for _ in range(ATTACK_INNER):
            r, t, _ = play_episode(env, agent, rng)
            coop_update(agent, t, r)
            batch.append((t, r))
        adv.update(batch)
    return adv


def learn_perturb(seed, agent):
    rng = np.random.default_rng(seed + 2000)
    env = MaskedHanabiSmall(HanabiSmallEnv())
    adv = LearnedPerturbation(NUM_ACTIONS, lr=0.01)
    for _ in range(ATTACK_OUTER):
        batch = []
        for _ in range(ATTACK_INNER):
            r, t, _ = play_episode(env, agent, rng, perturb=adv)
            coop_update(agent, t, r)
            batch.append((t, r))
        adv.update(batch)
    return adv


def entropy(items):
    if not items:
        return 0.0
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    n = sum(counts.values())
    return -sum((c / n) * math.log(c / n + 1e-12, 2) for c in counts.values())


def evaluate(env, agent, rng, mask_fn=None, perturb=None, episodes=EVAL):
    rewards, hint_events, actions, bad_discards = [], [], [], []
    for _ in range(episodes):
        if mask_fn is not None:
            env.set_mask(mask_fn)
        else:
            env.set_mask(None)
        r, t, metrics = play_episode(env, agent, rng, perturb=perturb)
        rewards.append(r)
        hint_events.extend(metrics["hints"])
        actions.extend(a for p, _s, a in t if p == 0)
        bad_discards.append(metrics["bad_discards"])
    hint_actions = [a for a in actions if a in (HINT_COLOR, HINT_RANK)]
    return {
        "reward": float(np.mean(rewards)),
        "hint_entropy": entropy(hint_events),
        "action_entropy": entropy(actions),
        "hint_rate": float(len(hint_actions) / max(1, len(actions))),
        "bad_discard_rate": float(np.mean(bad_discards)),
    }


def communication_removal(info_state, legal_actions, player):
    """Remove hint actions for both players, preserving at least one action."""
    filtered = [a for a in legal_actions if a not in (HINT_COLOR, HINT_RANK)]
    return filtered or legal_actions


def main():
    print("Hanabi-Small masking benchmark", flush=True)
    modes = ["none", "random", "perturb", "learned_removal", "comm_removal"]
    results = {m: [] for m in modes}
    for seed in SEEDS:
        env, agent, rng = train_base(seed)
        removal = learn_removal(seed, agent)
        perturb = learn_perturb(seed, agent)
        rand = random_mask(0, 0.3, np.random.default_rng(seed + 3000))
        configs = {
            "none": dict(mask_fn=None, perturb=None),
            "random": dict(mask_fn=rand, perturb=None),
            "perturb": dict(mask_fn=None, perturb=perturb),
            "learned_removal": dict(mask_fn=removal.mask_fn, perturb=None),
            "comm_removal": dict(mask_fn=communication_removal, perturb=None),
        }
        for mode, cfg in configs.items():
            vals = evaluate(env, agent, rng, **cfg)
            results[mode].append(vals)
            print(
                f"{mode:>8s} seed={seed} reward={vals['reward']:+.3f} "
                f"hintH={vals['hint_entropy']:.3f} actH={vals['action_entropy']:.3f} "
                f"badDisc={vals['bad_discard_rate']:.3f}",
                flush=True,
            )

    print("\nSummary", flush=True)
    for mode, vals in results.items():
        keys = vals[0].keys()
        summary = {k: np.mean([v[k] for v in vals]) for k in keys}
        print(
            f"{mode:>8s}: reward={summary['reward']:+.3f}, "
            f"hintH={summary['hint_entropy']:.3f}, actH={summary['action_entropy']:.3f}, "
            f"badDisc={summary['bad_discard_rate']:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
