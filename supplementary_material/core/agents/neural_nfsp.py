"""Neural NFSP agent: MLP average policy instead of tabular counts.

This is a "proper" NFSP with:
- Best-response: DQN (experience replay, target network)
- Average policy: supervised MLP trained on reservoir-sampled own actions
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class NeuralNFSPAgent:
    """NFSP with neural best-response (DQN) and neural average policy (MLP).

    Parameters
    ----------
    encoder : callable
        Maps info_state string to numpy float vector.
    input_dim : int
    num_actions : int
    eta : float
        Probability of playing best-response vs average policy.
    """

    def __init__(self, encoder, input_dim, num_actions, eta=0.1,
                 br_lr=1e-3, avg_lr=1e-3, epsilon=0.15,
                 buffer_size=20000, avg_buffer_size=50000, batch_size=64):
        self.encoder = encoder
        self.num_actions = num_actions
        self.eta = eta
        self.epsilon = epsilon
        self.batch_size = batch_size
        self._device = torch.device("cpu")

        # Best-response network (DQN)
        self._br_online = self._make_net(input_dim, num_actions)
        self._br_target = self._make_net(input_dim, num_actions)
        self._br_target.load_state_dict(self._br_online.state_dict())
        self._br_opt = optim.Adam(self._br_online.parameters(), lr=br_lr)
        self._br_buffer = deque(maxlen=buffer_size)

        # Average policy network
        self._avg_net = self._make_net(input_dim, num_actions)
        self._avg_opt = optim.Adam(self._avg_net.parameters(), lr=avg_lr)
        self._avg_buffer = deque(maxlen=avg_buffer_size)

        self._episode_count = 0
        self._using_br = False

    def _make_net(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, out_dim),
        ).to(self._device)

    def _encode(self, info_state):
        return torch.tensor(self.encoder(info_state), device=self._device).unsqueeze(0)

    def select_action(self, info_state, legal_actions, rng):
        self._using_br = rng.random() < self.eta

        if self._using_br:
            if rng.random() < self.epsilon:
                action = int(rng.choice(legal_actions))
            else:
                with torch.no_grad():
                    q = self._br_online(self._encode(info_state)).squeeze(0).cpu().numpy()
                masked = np.full(self.num_actions, -np.inf)
                for a in legal_actions: masked[a] = q[a]
                action = int(np.argmax(masked))
        else:
            with torch.no_grad():
                logits = self._avg_net(self._encode(info_state)).squeeze(0).cpu().numpy()
            masked = np.full(self.num_actions, -np.inf)
            for a in legal_actions: masked[a] = logits[a]
            exp = np.exp(masked - np.max(masked[masked > -np.inf]))
            exp[masked == -np.inf] = 0
            probs = exp / exp.sum()
            action = int(rng.choice(self.num_actions, p=probs))

        # Store in average policy buffer (reservoir sampling of own BR actions)
        if self._using_br:
            self._avg_buffer.append((info_state, action))

        return action

    def update(self, trajectory, reward_p0):
        for player, info_state, action in trajectory:
            reward = reward_p0 if player == 0 else -reward_p0
            self._br_buffer.append((info_state, action, reward))

        self._episode_count += 1
        if self._episode_count % 500 == 0:
            self._br_target.load_state_dict(self._br_online.state_dict())

        # BR update (DQN)
        if len(self._br_buffer) >= self.batch_size:
            batch = random.sample(list(self._br_buffer), self.batch_size)
            states = torch.tensor(np.array([self.encoder(s) for s, _, _ in batch]),
                                  device=self._device)
            actions = torch.tensor([a for _, a, _ in batch], dtype=torch.long,
                                   device=self._device)
            rewards = torch.tensor([r for _, _, r in batch], dtype=torch.float32,
                                   device=self._device)
            q_pred = self._br_online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            loss = nn.functional.mse_loss(q_pred, rewards)
            self._br_opt.zero_grad(); loss.backward(); self._br_opt.step()

        # Average policy update (supervised)
        if len(self._avg_buffer) >= self.batch_size:
            batch = random.sample(list(self._avg_buffer), self.batch_size)
            states = torch.tensor(np.array([self.encoder(s) for s, _ in batch]),
                                  device=self._device)
            actions = torch.tensor([a for _, a in batch], dtype=torch.long,
                                   device=self._device)
            logits = self._avg_net(states)
            loss = nn.functional.cross_entropy(logits, actions)
            self._avg_opt.zero_grad(); loss.backward(); self._avg_opt.step()

    def policy_probs(self, info_state, legal_actions):
        with torch.no_grad():
            logits = self._avg_net(self._encode(info_state)).squeeze(0).cpu().numpy()
        masked = np.full(self.num_actions, -np.inf)
        for a in legal_actions: masked[a] = logits[a]
        exp = np.exp(masked - np.max(masked[masked > -np.inf]))
        exp[masked == -np.inf] = 0
        probs = exp / exp.sum()
        return {a: float(probs[a]) for a in legal_actions}
