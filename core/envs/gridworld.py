"""Competitive Gridworld: simple non-poker adversarial masking test.

Two agents on a 5x5 grid. P0 (prey) tries to reach a goal.
P1 (predator) tries to catch P0. Both have 5 actions: UP/DOWN/LEFT/RIGHT/STAY.
Turn-based. Game ends when P0 reaches goal (P0 wins) or P1 catches P0 (P1 wins),
or after max_steps.

This is NOT poker. If adversarial masking works here too, the phenomenon
generalises beyond imperfect-information card games.
"""

import numpy as np

UP, DOWN, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4
NUM_ACTIONS = 5
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT", 4: "STAY"}
DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1), STAY: (0, 0)}


class CompetitiveGridworld:
    """5x5 grid, P0=prey (reaches goal=win), P1=predator (catches prey=win)."""

    def __init__(self, grid_size=5, max_steps=20):
        self.grid_size = grid_size
        self.max_steps = max_steps
        self._p0_pos = (0, 0)
        self._p1_pos = (4, 4)
        self._goal = (4, 0)
        self._step_count = 0
        self._done = False
        self._reward_p0 = 0.0
        self._current_player = 0

    def reset(self, rng=None, **kwargs):
        self._p0_pos = (0, 0)
        self._p1_pos = (self.grid_size - 1, self.grid_size - 1)
        self._goal = (self.grid_size - 1, 0)
        self._step_count = 0
        self._done = False
        self._reward_p0 = 0.0
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
        return [self._reward_p0, -self._reward_p0]

    def info_state(self, player):
        return f"{self._p0_pos}|{self._p1_pos}|{player}"

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

        self._step_count += 1

        if self._p0_pos == self._goal:
            self._done = True
            self._reward_p0 = 1.0
        elif self._p0_pos == self._p1_pos:
            self._done = True
            self._reward_p0 = -1.0
        elif self._step_count >= self.max_steps:
            self._done = True
            self._reward_p0 = -0.5  # timeout penalty for prey

        self._current_player = 1 - self._current_player


class MaskedGridworld:
    """Wraps CompetitiveGridworld with action masking."""

    def __init__(self, env):
        self.env = env
        self.mask_fn = None

    def set_mask(self, mask_fn):
        self.mask_fn = mask_fn

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        return self

    @property
    def current_player(self):
        return self.env.current_player

    def info_state(self, player):
        return self.env.info_state(player)

    def legal_actions(self):
        actions = self.env.legal_actions()
        if self.mask_fn is not None and not self.env.is_terminal:
            player = self.env.current_player
            info = self.env.info_state(player)
            masked = self.mask_fn(info, actions, player)
            if masked:
                return masked
        return actions

    @property
    def is_terminal(self):
        return self.env.is_terminal

    @property
    def returns(self):
        return self.env.returns

    def step(self, action):
        self.env.step(action)
