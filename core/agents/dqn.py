"""DQN agent for Kuhn / Leduc poker with experience replay and target network."""

import numpy as np
from collections import deque, defaultdict
import random

import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Feature encoders
# ---------------------------------------------------------------------------

def _kuhn_encoder(info_state: str) -> np.ndarray:
    """Kuhn info-state string -> float vector (dim 11).

    Layout: card one-hot(3) + history one-hot(2 actions * 4 positions = 8).
    """
    vec = np.zeros(11, dtype=np.float32)
    card = int(info_state[0])
    vec[card] = 1.0
    history = info_state[1:]
    for i, ch in enumerate(history):
        if i >= 4:
            break
        offset = 3 + i * 2
        vec[offset + (0 if ch == "p" else 1)] = 1.0
    return vec


def _leduc_encoder(info_state: str) -> np.ndarray:
    """Leduc info-state string -> float vector (dim 37).

    Layout: card one-hot(3) + community one-hot(3) + community_present(1)
            + history one-hot(3 actions * 10 positions = 30).
    """
    vec = np.zeros(37, dtype=np.float32)
    parts = info_state.split("|")
    card_idx = int(parts[0])
    vec[card_idx] = 1.0

    if len(parts) >= 3:
        comm = parts[1]
        if comm != "_":
            vec[3 + int(comm)] = 1.0
            vec[6] = 1.0
        history = parts[2]
    elif len(parts) == 2:
        history = parts[1]
    else:
        history = ""

    for i, ch in enumerate(history):
        if i >= 10:
            break
        offset = 7
        action_idx = {"f": 0, "c": 1, "r": 2}.get(ch, 0)
        vec[offset + i * 3 + action_idx] = 1.0
    return vec


def _leduc_n_encoder(num_ranks):
    """Create encoder for Leduc-N (parameterised by rank count).

    Layout: card one-hot(N) + community one-hot(N) + community_present(1)
            + history one-hot(3 actions * 10 positions = 30).
    Dim = 2*N + 1 + 30.
    """
    dim = 2 * num_ranks + 1 + 30

    def encoder(info_state: str) -> np.ndarray:
        vec = np.zeros(dim, dtype=np.float32)
        parts = info_state.split("|")
        card_idx = int(parts[0])
        vec[card_idx] = 1.0

        if len(parts) >= 3:
            comm = parts[1]
            if comm != "_":
                vec[num_ranks + int(comm)] = 1.0
                vec[2 * num_ranks] = 1.0
            history = parts[2]
        elif len(parts) == 2:
            history = parts[1]
        else:
            history = ""

        offset = 2 * num_ranks + 1
        for i, ch in enumerate(history):
            if i >= 10:
                break
            action_idx = {"f": 0, "c": 1, "r": 2}.get(ch, 0)
            vec[offset + i * 3 + action_idx] = 1.0
        return vec

    return encoder, dim


def get_encoder(game: str):
    """Factory returning (encoder_fn, input_dim) for a game name.

    Supported games: ``"kuhn"``, ``"leduc"``, ``"leduc5"``.
    """
    if game == "kuhn":
        return _kuhn_encoder, 11
    if game == "leduc":
        return _leduc_encoder, 37
    if game == "leduc5":
        return _leduc_n_encoder(5)
    if game == "leduc10":
        return _leduc_n_encoder(10)
    if game == "leduc20":
        return _leduc_n_encoder(20)
    if game.startswith("leduc") and game[5:].isdigit():
        return _leduc_n_encoder(int(game[5:]))
    raise ValueError(f"Unknown game: {game}")


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Simple 2-layer MLP for Q-value estimation."""

    def __init__(self, input_dim: int, num_actions: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """DQN agent with experience replay and a periodic target network.

    Parameters
    ----------
    game : str
        ``"kuhn"`` or ``"leduc"`` -- selects the feature encoder.
    num_actions : int
        Size of the action space.
    lr : float
        Learning rate for Adam.
    gamma : float
        Discount factor (typically 1.0 for episodic poker).
    epsilon_start, epsilon_end, epsilon_decay : float
        Epsilon-greedy schedule: ``eps = max(end, start * decay^episode)``.
    buffer_size : int
        Maximum replay-buffer capacity.
    batch_size : int
        Mini-batch size for each gradient step.
    target_update_freq : int
        Copy online network to target network every *N* episodes.
    """

    def __init__(
        self,
        game: str = "kuhn",
        num_actions: int = 2,
        lr: float = 1e-3,
        gamma: float = 1.0,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.999,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 500,
    ):
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self._encoder, input_dim = get_encoder(game)

        self._device = torch.device("cpu")
        self._online = QNetwork(input_dim, num_actions).to(self._device)
        self._target = QNetwork(input_dim, num_actions).to(self._device)
        self._target.load_state_dict(self._online.state_dict())
        self._optimizer = optim.Adam(self._online.parameters(), lr=lr)

        self._buffer: deque = deque(maxlen=buffer_size)
        self._episode_count = 0

    # -- helpers ----------------------------------------------------------

    def _encode(self, info_state: str) -> torch.Tensor:
        return torch.tensor(self._encoder(info_state), device=self._device).unsqueeze(0)

    # -- public interface (same as QLearningAgent) -------------------------

    def select_action(self, info_state: str, legal_actions: list, rng: np.random.Generator) -> int:
        """Epsilon-greedy action selection."""
        if rng.random() < self.epsilon:
            return int(rng.choice(legal_actions))
        with torch.no_grad():
            q_vals = self._online(self._encode(info_state)).squeeze(0).cpu().numpy()
        masked = np.full(self.num_actions, -np.inf)
        for a in legal_actions:
            masked[a] = q_vals[a]
        return int(np.argmax(masked))

    def update(self, trajectory: list, reward_p0: float) -> None:
        """Store episode transitions and run a mini-batch gradient step.

        ``trajectory`` is a list of ``(player, info_state, action)`` tuples
        collected during an episode.
        """
        for player, info_state, action in trajectory:
            reward = reward_p0 if player == 0 else -reward_p0
            self._buffer.append((info_state, action, reward))

        self._episode_count += 1
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        if self._episode_count % self.target_update_freq == 0:
            self._target.load_state_dict(self._online.state_dict())

        if len(self._buffer) < self.batch_size:
            return

        batch = random.sample(list(self._buffer), self.batch_size)
        states_np = np.array([self._encoder(s) for s, _, _ in batch])
        actions = np.array([a for _, a, _ in batch])
        rewards = np.array([r for _, _, r in batch], dtype=np.float32)

        states_t = torch.tensor(states_np, device=self._device)
        actions_t = torch.tensor(actions, device=self._device, dtype=torch.long)
        rewards_t = torch.tensor(rewards, device=self._device)

        q_pred = self._online(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            q_target = rewards_t + self.gamma * self._target(states_t).max(dim=1).values
            q_target = rewards_t  # terminal-only transitions: no next state

        loss = nn.functional.mse_loss(q_pred, q_target)
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def policy_probs(self, info_state: str, legal_actions: list) -> np.ndarray:
        """Return action probabilities (epsilon-greedy) for the adversary."""
        probs = np.zeros(self.num_actions)
        with torch.no_grad():
            q_vals = self._online(self._encode(info_state)).squeeze(0).cpu().numpy()
        masked = np.full(self.num_actions, -np.inf)
        for a in legal_actions:
            masked[a] = q_vals[a]
        best = int(np.argmax(masked))
        for a in legal_actions:
            probs[a] = self.epsilon / len(legal_actions)
        probs[best] += 1.0 - self.epsilon
        return probs
