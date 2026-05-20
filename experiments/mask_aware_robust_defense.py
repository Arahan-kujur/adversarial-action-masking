"""Mask-aware robust (C-MDP-style) defense baseline.

The victim is augmented with a binary availability vector for its own
actions. Both players use the same mask-aware Q-table; during training
we cycle through structured random masks so the victim experiences the
distribution of action-availability outcomes the reviewer asked about
under robust-MDP / C-MDP framing.
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adversary.mask_utils import evaluate_agent
from adversary.masking_policy import AdversarialMask
from core.agents.q_learning import QLearningAgent
from core.envs.kuhn_poker import KuhnPokerEnv, MaskedKuhnPoker
from core.envs.leduc_poker import LeducPokerEnv, MaskedLeducPoker, NUM_ACTIONS as LEDUC_ACTIONS
from core.training.selfplay import play_episode


SEEDS = list(range(5))
PRETRAIN = 6000
DEFENSE_EPISODES = 8000
ATTACK_OUTER = 12
ATTACK_INNER = 400
EVAL = 2000


def make_env(game):
    if game == "kuhn":
        return MaskedKuhnPoker(KuhnPokerEnv()), 2
    return MaskedLeducPoker(LeducPokerEnv()), LEDUC_ACTIONS


class MaskAwareAgent:
    """Q-learning that conditions on the per-state availability bitmap.

    Both players write to and read from the same mask-aware key namespace.
    """

    def __init__(self, num_actions, alpha=0.1, epsilon=0.15):
        self.num_actions = num_actions
        self.alpha = alpha
        self.epsilon = epsilon
        self.q = defaultdict(lambda: np.zeros(num_actions))

    def _key(self, info_state, legal_actions):
        bits = "".join("1" if a in legal_actions else "0" for a in range(self.num_actions))
        return f"{info_state}|m{bits}"

    def select_action(self, info_state, legal_actions, rng):
        key = self._key(info_state, legal_actions)
        if rng.random() < self.epsilon:
            return int(rng.choice(legal_actions))
        q_vals = self.q[key]
        return max(legal_actions, key=lambda a: q_vals[a])

    def update(self, trajectory, reward_p0):
        for player, key, action in trajectory:
            r = reward_p0 if player == 0 else -reward_p0
            self.q[key][action] += self.alpha * (r - self.q[key][action])


def play_mask_aware_episode(env, agent, rng):
    """Play one episode, storing per-step mask-aware keys for both players."""
    env.reset(rng=rng)
    traj = []
    while not env.is_terminal:
        p = env.current_player
        info = env.info_state(p)
        legal = env.legal_actions()
        if not legal:
            break
        key = agent._key(info, legal)
        action = agent.select_action(info, legal, rng)
        traj.append((p, key, action))
        env.step(action)
    return env.returns[0], traj


def random_structured_mask(num_actions, seed):
    rng = np.random.default_rng(seed)
    removed_action = int(rng.integers(0, num_actions))

    def mask_fn(info_state, legal_actions, player):
        if player != 0:
            return legal_actions
        filtered = [a for a in legal_actions if a != removed_action]
        return filtered or legal_actions

    return mask_fn


def train_mask_aware_robust(game, seed):
    rng = np.random.default_rng(seed)
    env, num_actions = make_env(game)
    agent = MaskAwareAgent(num_actions=num_actions, alpha=0.1, epsilon=0.15)
    for _ in range(PRETRAIN):
        r, t = play_mask_aware_episode(env, agent, rng)
        agent.update(t, r)
    masks = [random_structured_mask(num_actions, seed * 13 + i) for i in range(num_actions)]
    masks.append(None)
    for ep in range(DEFENSE_EPISODES):
        env.set_mask(masks[ep % len(masks)])
        r, t = play_mask_aware_episode(env, agent, rng)
        agent.update(t, r)
    env.set_mask(None)
    return env, agent, rng


def standard_victim(game, seed):
    rng = np.random.default_rng(seed)
    env, num_actions = make_env(game)
    agent = QLearningAgent(num_actions=num_actions, alpha=0.1, epsilon=0.15)
    for _ in range(PRETRAIN + DEFENSE_EPISODES):
        r, t = play_episode(env, agent, rng)
        agent.update(t, r)
    return env, agent, rng


def evaluate_mask_aware(game, agent, mask_fn, episodes, rng):
    env, _ = make_env(game)
    env.set_mask(mask_fn)
    rewards = []
    for _ in range(episodes):
        r, _ = play_mask_aware_episode(env, agent, rng)
        rewards.append(r)
    return float(np.mean(rewards))


def attack_and_eval_mask_aware(game, seed, agent):
    rng = np.random.default_rng(seed + 50000)
    env, num_actions = make_env(game)
    adv = AdversarialMask(target_player=0, num_actions=num_actions, lr=0.01)
    env.set_mask(adv.mask_fn)
    for _ in range(ATTACK_OUTER):
        batch = []
        for _ in range(ATTACK_INNER):
            r, t = play_mask_aware_episode(env, agent, rng)
            agent.update(t, r)
            batch.append((t, r))
        adv.update(batch)
    return evaluate_mask_aware(game, agent, adv.mask_fn, EVAL, rng)


def attack_and_eval_standard(game, seed, agent, env, rng):
    rng_eval = np.random.default_rng(seed + 50000)
    env_attack, num_actions = make_env(game)
    adv = AdversarialMask(target_player=0, num_actions=num_actions, lr=0.01)
    env_attack.set_mask(adv.mask_fn)
    for _ in range(ATTACK_OUTER):
        batch = []
        for _ in range(ATTACK_INNER):
            r, t = play_episode(env_attack, agent, rng_eval)
            agent.update(t, r)
            batch.append((t, r))
        adv.update(batch)
    return evaluate_agent(env_attack, agent, rng_eval, EVAL, mask_fn=adv.mask_fn)


def run(game):
    print(f"\n== {game.upper()} ==", flush=True)
    rows = {"standard": [], "mask_aware_robust": []}
    for seed in SEEDS:
        env_s, agent_s, rng_s = standard_victim(game, seed)
        clean_s = evaluate_agent(env_s, agent_s, rng_s, EVAL)
        atk_s = attack_and_eval_standard(game, seed, agent_s, env_s, rng_s)
        rows["standard"].append((clean_s, atk_s))
        print(
            f"  standard          seed={seed} clean={clean_s:+.3f} attacked={atk_s:+.3f}",
            flush=True,
        )

        env_m, agent_m, rng_m = train_mask_aware_robust(game, seed)
        clean_m = evaluate_mask_aware(game, agent_m, None, EVAL, rng_m)
        atk_m = attack_and_eval_mask_aware(game, seed, agent_m)
        rows["mask_aware_robust"].append((clean_m, atk_m))
        print(
            f"  mask_aware_robust seed={seed} clean={clean_m:+.3f} attacked={atk_m:+.3f}",
            flush=True,
        )
    for name, vals in rows.items():
        arr = np.asarray(vals, dtype=float)
        m = arr.mean(axis=0)
        ci = 1.96 * arr.std(axis=0) / np.sqrt(len(arr))
        print(
            f"  {name:>18s}: clean={m[0]:+.3f}+/-{ci[0]:.3f} attacked={m[1]:+.3f}+/-{ci[1]:.3f}",
            flush=True,
        )


def main():
    print("Mask-aware robust (C-MDP) defense", flush=True)
    for game in ["kuhn", "leduc"]:
        run(game)


if __name__ == "__main__":
    main()
