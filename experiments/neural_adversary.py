"""Neural-scale adversarial masking experiment for Leduc Poker.

Trains a neural adversary (small MLP) to learn which actions to remove
from a DQN victim, using bi-level optimization with REINFORCE.
Compares against no masking, random masking, and fixed masking.
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


# ---------------------------------------------------------------------------
# Neural Adversary
# ---------------------------------------------------------------------------

class NeuralAdversary(nn.Module):
    """MLP that outputs a distribution over which action to remove (or none).

    Output dimension is ``num_actions + 1``: indices 0..num_actions-1 each
    correspond to removing that game action, and the last index means
    "remove nothing".
    """

    def __init__(self, input_dim, num_actions):
        super().__init__()
        self.num_game_actions = num_actions
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_actions + 1),
        )
        self._encoder, _ = get_encoder("leduc")
        self._log_probs: list[torch.Tensor] = []
        self._rewards: list[float] = []
        self.collecting = True

    def forward(self, x):
        return torch.softmax(self.net(x), dim=-1)

    def mask_fn(self, info_state, legal_actions, player):
        """Mask function for MaskedLeducPoker.

        During training (``collecting=True``), samples removal and records
        log-probs for REINFORCE.  During eval, uses greedy argmax.
        """
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
        """Tag all unmatched log-probs with this episode's reward."""
        n = len(self._log_probs) - len(self._rewards)
        for _ in range(n):
            self._rewards.append(reward)

    def update(self, optimizer):
        """REINFORCE step: maximise -(victim reward)."""
        if not self._log_probs:
            return
        log_probs = torch.stack(self._log_probs)
        rewards = torch.FloatTensor(self._rewards)
        baseline = rewards.mean()
        # adversary wants to minimise victim reward -> gradient on -reward
        loss = -(log_probs * (-(rewards - baseline))).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self._log_probs.clear()
        self._rewards.clear()


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def pretrain_victim(env, agent, rng, num_episodes):
    """Self-play pre-training with no mask."""
    for _ in range(num_episodes):
        reward, traj = play_episode(env, agent, rng)
        agent.update(traj, reward)


def bilevel_train(env, victim, adversary, adv_optimizer, rng,
                  inner_episodes=1000, outer_iters=30):
    """Bi-level loop: victim adapts under mask, adversary updates via REINFORCE."""
    adversary.collecting = True
    env.set_mask(adversary.mask_fn)
    rewards_per_outer = []

    for outer in range(outer_iters):
        ep_rewards = []
        for _ in range(inner_episodes):
            reward, traj = play_episode(env, victim, rng)
            victim.update(traj, reward)
            adversary.record_reward(reward)
            ep_rewards.append(reward)

        adversary.update(adv_optimizer)
        mean_r = np.mean(ep_rewards)
        rewards_per_outer.append(mean_r)
        if (outer + 1) % 10 == 0:
            print(f"    Outer {outer+1:>3d}/{outer_iters}: "
                  f"victim reward = {mean_r:+.4f}")

    env.set_mask(None)
    adversary.collecting = False
    return rewards_per_outer


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def make_victim(seed_offset=0):
    """Create a fresh DQN victim for Leduc poker."""
    return DQNAgent(
        game="leduc", num_actions=NUM_ACTIONS,
        lr=1e-3, epsilon_start=1.0, epsilon_end=0.05,
        epsilon_decay=0.9999, buffer_size=20_000,
        batch_size=64, target_update_freq=500,
    )


def run_experiment(seeds=(42, 123, 456)):
    eval_episodes = 5000
    results = {
        "None": [],
        "Random (p=0.5)": [],
        "Fixed (raise)": [],
        "Neural Advers.": [],
    }

    for seed in seeds:
        print(f"\n{'='*55}")
        print(f"  Seed {seed}")
        print(f"{'='*55}")
        rng = np.random.default_rng(seed)
        torch.manual_seed(seed)

        # -- 1. Train shared DQN victim (30k self-play, no mask) -------------
        base_env = MaskedLeducPoker(LeducPokerEnv())
        victim = make_victim()
        print("  Pre-training DQN victim (30k self-play)...")
        pretrain_victim(base_env, victim, rng, 30_000)

        # -- 2. Evaluate: no mask -------------------------------------------
        r_none = evaluate_agent(base_env, victim, rng, eval_episodes)
        results["None"].append(r_none)
        print(f"  None:               {r_none:+.4f}")

        # -- 3. Evaluate: random masking -------------------------------------
        rand_env = MaskedLeducPoker(LeducPokerEnv())
        rand_mask = random_mask(target_player=0, remove_prob=0.5,
                                rng=np.random.default_rng(seed))
        r_rand = evaluate_agent(rand_env, copy.deepcopy(victim), rng,
                                eval_episodes, mask_fn=rand_mask)
        results["Random (p=0.5)"].append(r_rand)
        print(f"  Random (p=0.5):     {r_rand:+.4f}")

        # -- 4. Evaluate: fixed masking (remove RAISE) ----------------------
        fixed_env = MaskedLeducPoker(LeducPokerEnv())
        fixed_mask = fixed_removal(target_player=0, removed_action=2)
        r_fixed = evaluate_agent(fixed_env, copy.deepcopy(victim), rng,
                                 eval_episodes, mask_fn=fixed_mask)
        results["Fixed (raise)"].append(r_fixed)
        print(f"  Fixed (raise):      {r_fixed:+.4f}")

        # -- 5. Neural adversary (bi-level) ----------------------------------
        print("  Training neural adversary (bi-level)...")
        neural_env = MaskedLeducPoker(LeducPokerEnv())
        victim_neural = make_victim()
        pretrain_victim(neural_env, victim_neural, rng, 20_000)

        _, input_dim = get_encoder("leduc")
        adversary = NeuralAdversary(input_dim, NUM_ACTIONS)
        adv_optimizer = optim.Adam(adversary.parameters(), lr=1e-3)

        bilevel_train(neural_env, victim_neural, adversary, adv_optimizer, rng,
                      inner_episodes=1000, outer_iters=30)

        r_neural = evaluate_agent(neural_env, victim_neural, rng,
                                  eval_episodes, mask_fn=adversary.mask_fn)
        results["Neural Advers."].append(r_neural)
        print(f"  Neural adversary:   {r_neural:+.4f}")

    # -- Summary table -------------------------------------------------------
    print(f"\n{'='*55}")
    print(f"  RESULTS (averaged over {len(seeds)} seeds)")
    print(f"{'='*55}")
    hdr = "Leduc DQN Victim Reward"
    print(f"  {'Strategy':<16s} | {hdr:>24s}")
    print(f"  {'-'*16}-+-{'-'*24}")
    for strategy, vals in results.items():
        mean = np.mean(vals)
        std = np.std(vals)
        print(f"  {strategy:<16s} | {mean:+.4f} +/- {std:.4f}")
    print()


if __name__ == "__main__":
    run_experiment()
