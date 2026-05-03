"""Neural-scale adversarial masking: DQN victim in Leduc Poker.

Reduced scale for tractability: 5k pretrain, 10 outer x 200 inner bi-level.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from core.envs.leduc_poker import LeducPokerEnv, MaskedLeducPoker, NUM_ACTIONS
from core.agents.dqn import DQNAgent, get_encoder
from core.training.selfplay import play_episode
from adversary.mask_utils import evaluate_agent
from adversary.masking_policy import random_mask, fixed_removal


class NeuralAdversary(nn.Module):
    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.num_game_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions + 1),
        )
        self._encoder, _ = get_encoder("leduc")
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


PRETRAIN = 2000
INNER_EPS = 100
OUTER_ITERS = 10
EVAL_EPS = 1000
SEEDS = [42, 123, 456]


def make_victim():
    return DQNAgent(
        game="leduc", num_actions=NUM_ACTIONS,
        lr=1e-3, epsilon_start=0.3, epsilon_end=0.05,
        epsilon_decay=0.998, buffer_size=5000,
        batch_size=32, target_update_freq=100,
    )


def pretrain(env, victim, rng, n):
    import sys
    for i in range(n):
        r, t = play_episode(env, victim, rng)
        victim.update(t, r)
        if (i + 1) % 500 == 0:
            print(f"    pretrain {i+1}/{n}", flush=True)


def run_experiment():
    results = {"None": [], "Random (p=0.5)": [], "Fixed (raise)": [], "Neural Advers.": []}

    for seed in SEEDS:
        print(f"\n{'='*55}\n  Seed {seed}\n{'='*55}")
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        # 1. Pretrain base victim
        base_env = MaskedLeducPoker(LeducPokerEnv())
        victim = make_victim()
        print("  Pretraining...")
        pretrain(base_env, victim, rng, PRETRAIN)

        # 2. No mask
        r_none = evaluate_agent(base_env, victim, rng, EVAL_EPS)
        results["None"].append(r_none)
        print(f"  None:             {r_none:+.4f}", flush=True)

        # 3. Random (eval only, no retraining under mask)
        rand_env = MaskedLeducPoker(LeducPokerEnv())
        rm = random_mask(target_player=0, remove_prob=0.5, rng=np.random.default_rng(seed))
        r_rand = evaluate_agent(rand_env, copy.deepcopy(victim), rng, EVAL_EPS, mask_fn=rm)
        results["Random (p=0.5)"].append(r_rand)
        print(f"  Random (p=0.5):   {r_rand:+.4f}", flush=True)

        # 4. Fixed (remove RAISE, eval only)
        fixed_env = MaskedLeducPoker(LeducPokerEnv())
        fm = fixed_removal(target_player=0, removed_action=2)
        r_fixed = evaluate_agent(fixed_env, copy.deepcopy(victim), rng, EVAL_EPS, mask_fn=fm)
        results["Fixed (raise)"].append(r_fixed)
        print(f"  Fixed (raise):    {r_fixed:+.4f}", flush=True)

        # 5. Neural adversary
        print("  Training neural adversary...")
        neural_env = MaskedLeducPoker(LeducPokerEnv())
        v_neural = make_victim()
        pretrain(neural_env, v_neural, rng, PRETRAIN)

        _, input_dim = get_encoder("leduc")
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
                print(f"    outer {outer+1}/{OUTER_ITERS}: victim={np.mean(ep_r):+.4f}", flush=True)

        neural_env.set_mask(None)
        adv.collecting = False
        r_neural = evaluate_agent(neural_env, v_neural, rng, EVAL_EPS, mask_fn=adv.mask_fn)
        results["Neural Advers."].append(r_neural)
        print(f"  Neural adversary: {r_neural:+.4f}")

    print(f"\n{'='*55}\n  RESULTS ({len(SEEDS)} seeds)\n{'='*55}")
    print(f"  {'Strategy':<16s} | {'Leduc DQN Victim':>20s}")
    print(f"  {'-'*16}-+-{'-'*20}")
    for strat, vals in results.items():
        m, s = np.mean(vals), np.std(vals)
        print(f"  {strat:<16s} | {m:+.4f} +/- {s:.4f}")


if __name__ == "__main__":
    run_experiment()
