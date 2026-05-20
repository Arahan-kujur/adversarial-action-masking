"""Hanabi-V2 (3 colors, 5 ranks, hand 3) action removal benchmark."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adversary.masking_policy import AdversarialMask, random_mask
from core.agents.q_learning import QLearningAgent
from core.envs.hanabi_small import (
    HanabiV2Env,
    MaskedHanabiV2,
    V2_NUM_ACTIONS,
    V2_HINT_COLOR_OFFSET,
    V2_HINT_RANK_OFFSET,
    V2_COLORS,
    V2_RANKS,
)


SEEDS = [0, 1, 2]
PRETRAIN = 8000
ATTACK_OUTER = 12
ATTACK_INNER = 400
EVAL = 1500


class LearnedPerturbation:
    def __init__(self, num_actions=V2_NUM_ACTIONS, lr=0.01):
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
        if not legal:
            break
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
    env = MaskedHanabiV2(HanabiV2Env())
    agent = QLearningAgent(num_actions=V2_NUM_ACTIONS, alpha=0.06, epsilon=0.2)
    for _ in range(PRETRAIN):
        r, t, _ = play_episode(env, agent, rng)
        coop_update(agent, t, r)
    return env, agent, rng


def learn_removal(seed, agent):
    rng = np.random.default_rng(seed + 1000)
    env = MaskedHanabiV2(HanabiV2Env())
    adv = AdversarialMask(target_player=0, num_actions=V2_NUM_ACTIONS, lr=0.01)
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
    env = MaskedHanabiV2(HanabiV2Env())
    adv = LearnedPerturbation(V2_NUM_ACTIONS, lr=0.01)
    for _ in range(ATTACK_OUTER):
        batch = []
        for _ in range(ATTACK_INNER):
            r, t, _ = play_episode(env, agent, rng, perturb=adv)
            coop_update(agent, t, r)
            batch.append((t, r))
        adv.update(batch)
    return adv


def communication_removal(info_state, legal_actions, player):
    if player != 0:
        return legal_actions
    filtered = [a for a in legal_actions if a < V2_HINT_COLOR_OFFSET]
    return filtered or legal_actions


def color_only(info_state, legal_actions, player):
    if player != 0:
        return legal_actions
    filtered = [a for a in legal_actions if a < V2_HINT_RANK_OFFSET or a >= V2_HINT_RANK_OFFSET + V2_RANKS]
    return filtered or legal_actions


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
        env.set_mask(mask_fn)
        r, t, metrics = play_episode(env, agent, rng, perturb=perturb)
        rewards.append(r)
        hint_events.extend(metrics["hints"])
        actions.extend(a for p, _s, a in t if p == 0)
        bad_discards.append(metrics["bad_discards"])
    return {
        "reward": float(np.mean(rewards)),
        "hint_entropy": entropy(hint_events),
        "action_entropy": entropy(actions),
        "bad_discard_rate": float(np.mean(bad_discards)),
    }


def main():
    print("Hanabi-V2 (3 colors x 5 ranks, hand size 3) masking benchmark", flush=True)
    print(f"Action space size: {V2_NUM_ACTIONS}", flush=True)
    modes = ["none", "random", "perturb", "learned_removal", "comm_removal", "rank_hints_removed"]
    results = {m: [] for m in modes}
    for seed in SEEDS:
        env, agent, rng = train_base(seed)
        removal = learn_removal(seed, agent)
        perturb = learn_perturb(seed, agent)
        rand = random_mask(0, 0.25, np.random.default_rng(seed + 3000))
        configs = {
            "none": dict(mask_fn=None, perturb=None),
            "random": dict(mask_fn=rand, perturb=None),
            "perturb": dict(mask_fn=None, perturb=perturb),
            "learned_removal": dict(mask_fn=removal.mask_fn, perturb=None),
            "comm_removal": dict(mask_fn=communication_removal, perturb=None),
            "rank_hints_removed": dict(mask_fn=color_only, perturb=None),
        }
        for mode, cfg in configs.items():
            vals = evaluate(env, agent, rng, **cfg)
            results[mode].append(vals)
            print(
                f"{mode:>20s} seed={seed} reward={vals['reward']:+.3f} "
                f"hintH={vals['hint_entropy']:.3f} actH={vals['action_entropy']:.3f} "
                f"badDisc={vals['bad_discard_rate']:.3f}",
                flush=True,
            )

    print("\nSummary", flush=True)
    for mode, vals in results.items():
        arr = {k: [v[k] for v in vals] for k in vals[0].keys()}
        print(
            f"{mode:>20s}: reward={np.mean(arr['reward']):+.3f}, "
            f"hintH={np.mean(arr['hint_entropy']):.3f}, "
            f"actH={np.mean(arr['action_entropy']):.3f}, "
            f"badDisc={np.mean(arr['bad_discard_rate']):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
