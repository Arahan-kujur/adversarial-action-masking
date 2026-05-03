"""Resource Collection Game: second non-poker cross-domain test.

Two agents compete to collect resources on a 4x4 grid.
- 4 resources spawn at random positions each episode
- Players alternate turns, 4 actions (UP/DOWN/LEFT/RIGHT)
- Collecting a resource = +1 for that player
- Game ends when all resources collected or max_steps reached
- P0 reward = P0_collected - P1_collected

Different from gridworld (goal-seeking) and poker (cards/betting).
Tests adversarial masking in a resource-competition setting.
"""

import numpy as np

UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
NUM_ACTIONS = 4
DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}


class ResourceCollectionEnv:
    def __init__(self, grid_size=4, num_resources=4, max_steps=16):
        self.grid_size = grid_size
        self.num_resources = num_resources
        self.max_steps = max_steps
        self._p0_pos = (0, 0)
        self._p1_pos = (grid_size - 1, grid_size - 1)
        self._resources = set()
        self._collected = [0, 0]
        self._step_count = 0
        self._done = False
        self._current_player = 0

    def reset(self, rng=None, **kwargs):
        self._p0_pos = (0, 0)
        self._p1_pos = (self.grid_size - 1, self.grid_size - 1)
        positions = set()
        positions.add(self._p0_pos)
        positions.add(self._p1_pos)
        self._resources = set()
        while len(self._resources) < self.num_resources:
            r = int(rng.integers(0, self.grid_size))
            c = int(rng.integers(0, self.grid_size))
            if (r, c) not in positions:
                self._resources.add((r, c))
                positions.add((r, c))
        self._collected = [0, 0]
        self._step_count = 0
        self._done = False
        self._current_player = 0
        return self

    @property
    def current_player(self):
        return self._current_player

    @property
    def is_terminal(self):
        return self._done

    @property
    def returns(self):
        r = self._collected[0] - self._collected[1]
        return [r, -r]

    def info_state(self, player):
        res_str = "".join("1" if (r, c) in self._resources else "0"
                          for r in range(self.grid_size)
                          for c in range(self.grid_size))
        return f"{self._p0_pos}|{self._p1_pos}|{res_str}|{player}"

    def legal_actions(self):
        if self._done:
            return []
        return list(range(NUM_ACTIONS))

    def step(self, action):
        assert not self._done
        pos = self._p0_pos if self._current_player == 0 else self._p1_pos
        dr, dc = DELTAS[action]
        new_r = max(0, min(self.grid_size - 1, pos[0] + dr))
        new_c = max(0, min(self.grid_size - 1, pos[1] + dc))
        new_pos = (new_r, new_c)

        if self._current_player == 0:
            self._p0_pos = new_pos
        else:
            self._p1_pos = new_pos

        if new_pos in self._resources:
            self._resources.remove(new_pos)
            self._collected[self._current_player] += 1

        self._step_count += 1
        if not self._resources or self._step_count >= self.max_steps:
            self._done = True

        self._current_player = 1 - self._current_player


class MaskedResourceCollection:
    def __init__(self, env):
        self.env = env
        self.mask_fn = None

    def set_mask(self, mask_fn):
        self.mask_fn = mask_fn

    def reset(self, **kwargs):
        self.env.reset(**kwargs); return self

    @property
    def current_player(self):
        return self.env.current_player

    def info_state(self, player):
        return self.env.info_state(player)

    def legal_actions(self):
        actions = self.env.legal_actions()
        if self.mask_fn and not self.env.is_terminal:
            p = self.env.current_player
            info = self.env.info_state(p)
            m = self.mask_fn(info, actions, p)
            if m: return m
        return actions

    @property
    def is_terminal(self):
        return self.env.is_terminal

    @property
    def returns(self):
        return self.env.returns

    def step(self, action):
        self.env.step(action)
