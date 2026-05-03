"""Leduc-5 scale experiment: DQN victim + neural adversary.

Leduc-5: 5 ranks x 2 suits = 10 cards, 3 actions, 2 rounds.
~300+ P0 info states. This is genuine function-approximation territory.

First: enumerate info states to report the actual count.
Then: run DQN + neural adversary, compare against baselines.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.envs.leduc_n import LeducNPokerEnv, MaskedLeducNPoker, NUM_ACTIONS
from core.agents.dqn import DQNAgent, get_encoder
from core.training.selfplay import play_episode
from adversary.mask_utils import evaluate_agent
from adversary.masking_policy import random_mask, fixed_removal


class NeuralAdversary(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.num_game_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions + 1),
        )
        self._encoder, _ = get_encoder("leduc5")
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
            remove_idx = dist.sample()
            self._log_probs.append(dist.log_prob(remove_idx))
            remove_idx = int(remove_idx.item())
        else:
            with torch.no_grad():
                probs = self.forward(features)[0]
            remove_idx = int(probs.argmax().item())
        if remove_idx < self.num_game_actions and remove_idx in legal_actions:
            filtered = [a for a in legal_actions if a != remove_idx]
            if filtered:
                return filtered
        return legal_actions

    def record_reward(self, reward):
        n = len(self._log_probs) - len(self._rewards)
        for _ in range(n):
            self._rewards.append(reward)

    def update(self, optimizer):
        if not self._log_probs:
            return
        log_probs = torch.stack(self._log_probs)
        rewards = torch.FloatTensor(self._rewards)
        baseline = rewards.mean()
        loss = -(log_probs * (-(rewards - baseline))).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self._log_probs.clear()
        self._rewards.clear()


PRETRAIN = 20000
INNER_EPS = 500
OUTER_ITERS = 20
EVAL_EPS = 3000
SEEDS = list(range(5))


def make_victim():
    return DQNAgent(
        game="leduc5", num_actions=NUM_ACTIONS,
        lr=1e-3, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay=0.9999, buffer_size=20000,
        batch_size=64, target_update_freq=500,
    )


def pretrain(env, victim, rng, n):
    for i in range(n):
        r, t = play_episode(env, victim, rng)
        victim.update(t, r)
        if (i + 1) % 5000 == 0:
            print(f"      pretrain {i+1}/{n}", flush=True)


def count_info_states(env, agent, rng, n=10000):
    """Count unique P0 info states seen over n episodes."""
    states = set()
    for _ in range(n):
        env.reset(rng=rng)
        while not env.is_terminal:
            player = env.current_player
            info = env.info_state(player)
            if player == 0:
                states.add(info)
            legal = env.legal_actions()
            action = agent.select_action(info, legal, rng)
            env.step(action)
    return states


def run_experiment():
    print("=" * 65, flush=True)
    print("  LEDUC-5 SCALE EXPERIMENT", flush=True)
    print("  5 ranks x 2 suits = 10 cards, 3 actions, 2 rounds", flush=True)
    print("=" * 65, flush=True)

    # Count info states first
    rng = np.random.default_rng(0)
    env = MaskedLeducNPoker(LeducNPokerEnv(num_ranks=5))
    agent = make_victim()
    pretrain(env, agent, rng, 5000)
    states = count_info_states(env, agent, rng, 20000)
    print(f"\n  Unique P0 info states observed: {len(states)}", flush=True)

    results = {"None": [], "Random": [], "Fixed": [], "Neural Adv": []}

    for seed in SEEDS:
        print(f"\n{'='*65}\n  Seed {seed}\n{'='*65}", flush=True)
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        base_env = MaskedLeducNPoker(LeducNPokerEnv(num_ranks=5))
        victim = make_victim()
        print("  Pretraining DQN...", flush=True)
        pretrain(base_env, victim, rng, PRETRAIN)

        # No mask
        r_none = evaluate_agent(base_env, victim, rng, EVAL_EPS)
        results["None"].append(r_none)
        print(f"  None:        {r_none:+.4f}", flush=True)

        # Random
        rand_env = MaskedLeducNPoker(LeducNPokerEnv(num_ranks=5))
        rm = random_mask(target_player=0, remove_prob=0.5,
                         rng=np.random.default_rng(seed + 99))
        r_rand = evaluate_agent(rand_env, copy.deepcopy(victim), rng,
                                EVAL_EPS, mask_fn=rm)
        results["Random"].append(r_rand)
        print(f"  Random:      {r_rand:+.4f}", flush=True)

        # Fixed (remove RAISE)
        fixed_env = MaskedLeducNPoker(LeducNPokerEnv(num_ranks=5))
        fm = fixed_removal(target_player=0, removed_action=2)
        r_fixed = evaluate_agent(fixed_env, copy.deepcopy(victim), rng,
                                 EVAL_EPS, mask_fn=fm)
        results["Fixed"].append(r_fixed)
        print(f"  Fixed:       {r_fixed:+.4f}", flush=True)

        # Neural adversary
        print("  Neural adversary training...", flush=True)
        neural_env = MaskedLeducNPoker(LeducNPokerEnv(num_ranks=5))
        v_neural = make_victim()
        pretrain(neural_env, v_neural, rng, PRETRAIN)

        _, input_dim = get_encoder("leduc5")
        adv = NeuralAdversary(input_dim, NUM_ACTIONS)
        adv_opt = optim.Adam(adv.parameters(), lr=1e-3)
        adv.collecting = True
        neural_env.set_mask(adv.mask_fn)

        for outer in range(OUTER_ITERS):
            ep_r = []
            for _ in range(INNER_EPS):
                r, t = play_episode(neural_env, v_neural, rng)
                v_neural.update(t, r)
                adv.record_reward(r)
                ep_r.append(r)
            adv.update(adv_opt)
            if (outer + 1) % 5 == 0:
                print(f"      outer {outer+1}/{OUTER_ITERS}: "
                      f"victim={np.mean(ep_r):+.4f}", flush=True)

        neural_env.set_mask(None)
        adv.collecting = False
        r_neural = evaluate_agent(neural_env, v_neural, rng, EVAL_EPS,
                                  mask_fn=adv.mask_fn)
        results["Neural Adv"].append(r_neural)
        print(f"  Neural Adv:  {r_neural:+.4f}", flush=True)

    # Summary
    r_min, r_max = -17.0, 17.0  # Leduc-5 max pot is larger
    print(f"\n{'='*65}")
    print(f"  LEDUC-5 DQN RESULTS ({len(SEEDS)} seeds)")
    print(f"{'='*65}")
    print(f"  {'Strategy':<12s} | {'Raw Reward':>16s} | {'Normalized':>12s}")
    print(f"  {'-'*12}-+-{'-'*16}-+-{'-'*12}")
    for strat in results:
        vals = results[strat]
        m = np.mean(vals)
        ci = 1.96 * np.std(vals) / np.sqrt(len(vals))
        norm = (m - r_min) / (r_max - r_min)
        print(f"  {strat:<12s} | {m:+.3f} +/- {ci:.3f} | {norm:.3f}")
    print(flush=True)


if __name__ == "__main__":
    run_experiment()
