"""Leduc-30 intermediate scale check with DQN victim and neural adversary.

Fills in the scaling regression between Leduc-20 (5,531 states) and
Leduc-50 (~13,000 states) so the trend is established across 6 sizes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from adversary.mask_utils import evaluate_agent
from adversary.masking_policy import random_mask
from core.agents.dqn import DQNAgent, get_encoder
from core.envs.leduc_n import LeducNPokerEnv, MaskedLeducNPoker, NUM_ACTIONS
from core.training.selfplay import play_episode


NUM_RANKS = 30
GAME = "leduc30"
PRETRAIN = 30000
INNER = 500
OUTER = 25
EVAL = 3000
SEEDS = list(range(5))


class NeuralAdversary(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.num_game_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions + 1),
        )
        self._encoder, _ = get_encoder(GAME)
        self._log_probs = []
        self._rewards = []
        self.collecting = True

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

    def mask_fn(self, info_state, legal_actions, player):
        if player != 0 or len(legal_actions) <= 1:
            return legal_actions
        features = torch.FloatTensor(self._encoder(info_state)).unsqueeze(0)
        if self.collecting:
            probs = self.forward(features)[0]
            dist = torch.distributions.Categorical(probs)
            idx = dist.sample()
            self._log_probs.append(dist.log_prob(idx))
            idx = int(idx.item())
        else:
            with torch.no_grad():
                idx = int(self.forward(features)[0].argmax().item())
        if idx < self.num_game_actions and idx in legal_actions:
            filtered = [a for a in legal_actions if a != idx]
            if filtered:
                return filtered
        return legal_actions

    def record_reward(self, reward):
        n = len(self._log_probs) - len(self._rewards)
        self._rewards.extend([reward] * n)

    def update(self, opt):
        if not self._log_probs:
            return
        log_probs = torch.stack(self._log_probs)
        rewards = torch.FloatTensor(self._rewards)
        loss = -(log_probs * (-(rewards - rewards.mean()))).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        self._log_probs.clear()
        self._rewards.clear()


def make_env():
    return MaskedLeducNPoker(LeducNPokerEnv(num_ranks=NUM_RANKS))


def make_victim():
    return DQNAgent(
        game=GAME,
        num_actions=NUM_ACTIONS,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.99995,
        buffer_size=50000,
        batch_size=128,
        target_update_freq=1000,
    )


def pretrain(env, victim, rng):
    for _ in range(PRETRAIN):
        r, t = play_episode(env, victim, rng)
        victim.update(t, r)


def count_states(env, agent, rng, episodes=10000):
    states = set()
    for _ in range(episodes):
        env.reset(rng=rng)
        while not env.is_terminal:
            p = env.current_player
            info = env.info_state(p)
            if p == 0:
                states.add(info)
            a = agent.select_action(info, env.legal_actions(), rng)
            env.step(a)
    return len(states)


def main():
    print("Leduc-30 DQN scale check (5 seeds, full episodes)", flush=True)
    results = {"none": [], "random": [], "neural_adv": [], "states": []}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        env = make_env()
        victim = make_victim()
        pretrain(env, victim, rng)
        states = count_states(env, victim, rng)
        none = evaluate_agent(env, victim, rng, EVAL)
        rand = evaluate_agent(
            env,
            victim,
            rng,
            EVAL,
            mask_fn=random_mask(0, 0.3, np.random.default_rng(seed + 100)),
        )

        adv_env = make_env()
        adv = NeuralAdversary(get_encoder(GAME)[1], NUM_ACTIONS)
        opt = optim.Adam(adv.parameters(), lr=1e-3)
        adv_env.set_mask(adv.mask_fn)
        for _ in range(OUTER):
            for _ in range(INNER):
                r, t = play_episode(adv_env, victim, rng)
                victim.update(t, r)
                adv.record_reward(r)
            adv.update(opt)
        adv.collecting = False
        adv_reward = evaluate_agent(adv_env, victim, rng, EVAL, mask_fn=adv.mask_fn)

        results["states"].append(states)
        results["none"].append(none)
        results["random"].append(rand)
        results["neural_adv"].append(adv_reward)
        print(
            f"seed={seed} states={states} none={none:+.3f} random={rand:+.3f} adv={adv_reward:+.3f}",
            flush=True,
        )

    for key, vals in results.items():
        vals = np.asarray(vals, dtype=float)
        print(f"{key}: {vals.mean():+.3f} +/- {1.96 * vals.std() / np.sqrt(len(vals)):.3f}", flush=True)


if __name__ == "__main__":
    main()
