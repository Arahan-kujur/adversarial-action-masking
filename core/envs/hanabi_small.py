"""Hanabi-Small cooperative benchmark with action masking.

This is a deliberately tiny Hanabi variant (2 colors, 2 ranks, hand size 2)
that keeps the existing tabular self-play infrastructure tractable while
preserving the key features: hidden self cards, public partner cards, hints,
and convention-dependent coordination.
"""
from __future__ import annotations

import numpy as np


PLAY0, PLAY1, DISCARD0, DISCARD1, HINT_COLOR, HINT_RANK = range(6)
NUM_ACTIONS = 6
ACTION_NAMES = {
    PLAY0: "PLAY0",
    PLAY1: "PLAY1",
    DISCARD0: "DISCARD0",
    DISCARD1: "DISCARD1",
    HINT_COLOR: "HINT_COLOR",
    HINT_RANK: "HINT_RANK",
}


class HanabiSmallEnv:
    def __init__(self, max_steps=16):
        self.max_steps = max_steps
        self.deck = []
        self.hands = [[], []]
        self.fireworks = [0, 0]
        self.discard = []
        self.hints = 3
        self.lives = 2
        self.step_count = 0
        self.current_player_id = 0
        self.done = False
        self.score = 0.0
        self.last_hint = [("none", -1), ("none", -1)]
        self.metrics = {
            "hints": [],
            "bad_discards": 0,
            "plays": [],
        }

    def reset(self, rng=None, **kwargs):
        rng = rng or np.random.default_rng()
        # Two copies of each color/rank card.
        self.deck = [(c, r) for c in range(2) for r in range(2) for _ in range(2)]
        rng.shuffle(self.deck)
        self.hands = [[], []]
        self.fireworks = [0, 0]
        self.discard = []
        self.hints = 3
        self.lives = 2
        self.step_count = 0
        self.current_player_id = 0
        self.done = False
        self.score = 0.0
        self.last_hint = [("none", -1), ("none", -1)]
        self.metrics = {"hints": [], "bad_discards": 0, "plays": []}
        for player in [0, 1]:
            for _ in range(2):
                self._draw(player)
        return self

    @property
    def current_player(self):
        return self.current_player_id

    @property
    def is_terminal(self):
        return self.done

    @property
    def returns(self):
        return [self.score, self.score]

    def legal_actions(self):
        if self.done:
            return []
        legal = [PLAY0, PLAY1, DISCARD0, DISCARD1]
        if self.hints > 0:
            legal.extend([HINT_COLOR, HINT_RANK])
        return legal

    def info_state(self, player):
        partner = 1 - player
        partner_cards = ",".join(f"{c}{r}" for c, r in self.hands[partner])
        own_hint = self.last_hint[player]
        fireworks = "".join(str(x) for x in self.fireworks)
        disc = "".join(f"{c}{r}" for c, r in self.discard[-4:])
        return f"p{player}|partner={partner_cards}|fw={fireworks}|h={self.hints}|hint={own_hint[0]}{own_hint[1]}|d={disc}"

    def step(self, action):
        assert not self.done
        player = self.current_player_id
        if action in (PLAY0, PLAY1):
            idx = action - PLAY0
            self._play(player, idx)
        elif action in (DISCARD0, DISCARD1):
            idx = action - DISCARD0
            self._discard(player, idx)
        elif action == HINT_COLOR:
            self._hint(player, "color")
        elif action == HINT_RANK:
            self._hint(player, "rank")
        else:
            raise ValueError(action)

        self.step_count += 1
        if self.lives <= 0 or self.score >= 4 or self.step_count >= self.max_steps:
            self.done = True
        self.current_player_id = 1 - self.current_player_id

    def _draw(self, player):
        if self.deck:
            self.hands[player].append(self.deck.pop())

    def _replace(self, player, idx):
        if 0 <= idx < len(self.hands[player]):
            self.hands[player].pop(idx)
            self._draw(player)

    def _play(self, player, idx):
        if idx >= len(self.hands[player]):
            return
        card = self.hands[player][idx]
        color, rank = card
        self.metrics["plays"].append(card)
        if rank == self.fireworks[color]:
            self.fireworks[color] += 1
            self.score += 1.0
        else:
            self.lives -= 1
            self.discard.append(card)
        self._replace(player, idx)

    def _discard(self, player, idx):
        if idx >= len(self.hands[player]):
            return
        card = self.hands[player][idx]
        color, rank = card
        if rank == self.fireworks[color]:
            self.metrics["bad_discards"] += 1
        self.discard.append(card)
        self.hints = min(3, self.hints + 1)
        self._replace(player, idx)

    def _hint(self, player, hint_type):
        partner = 1 - player
        if self.hints <= 0 or not self.hands[partner]:
            return
        self.hints -= 1
        value = self.hands[partner][0][0 if hint_type == "color" else 1]
        self.last_hint[partner] = (hint_type, value)
        self.metrics["hints"].append((hint_type, value))


class MaskedHanabiSmall:
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

    @property
    def is_terminal(self):
        return self.env.is_terminal

    @property
    def returns(self):
        return self.env.returns

    @property
    def metrics(self):
        return self.env.metrics

    def info_state(self, player):
        return self.env.info_state(player)

    def legal_actions(self):
        actions = self.env.legal_actions()
        if self.mask_fn and not self.env.is_terminal:
            player = self.env.current_player
            info = self.env.info_state(player)
            masked = self.mask_fn(info, actions, player)
            if masked:
                return masked
        return actions

    def step(self, action):
        self.env.step(action)


# ===========================================================================
# Hanabi-V2: closer to the real Hanabi specification.
#
# - 3 colors, 5 ranks, standard rank distribution [3, 2, 2, 2, 1] per color.
# - Hand size 3.
# - 3 color hints + 5 rank hints addressed at the partner (one action each).
# - 3 play + 3 discard slot-indexed actions.
# - Total 14 actions; partner cards are observable, own cards are hidden but
#   accumulate the union of received hints for each slot.
# - Mistakes cost a life; running out of lives ends the game with current
#   score; reaching the maximum 15 (3 colors x 5 ranks) ends with a win.
# ===========================================================================

V2_COLORS = 3
V2_RANKS = 5
V2_HAND_SIZE = 3
V2_RANK_DIST = [3, 2, 2, 2, 1]
V2_INITIAL_HINTS = 4
V2_MAX_HINTS = 8
V2_LIVES = 3
V2_NUM_ACTIONS = V2_HAND_SIZE * 2 + V2_COLORS + V2_RANKS

V2_PLAY_OFFSET = 0
V2_DISCARD_OFFSET = V2_HAND_SIZE
V2_HINT_COLOR_OFFSET = 2 * V2_HAND_SIZE
V2_HINT_RANK_OFFSET = V2_HINT_COLOR_OFFSET + V2_COLORS


class HanabiV2Env:
    """Larger Hanabi variant with color/rank hint actions and proper rank distribution."""

    def __init__(self, max_steps=40):
        self.max_steps = max_steps
        self.deck = []
        self.hands = [[], []]
        self.knowledge = [[], []]
        self.fireworks = [0] * V2_COLORS
        self.discard = []
        self.hints = V2_INITIAL_HINTS
        self.lives = V2_LIVES
        self.step_count = 0
        self.current_player_id = 0
        self.done = False
        self.score = 0.0
        self.metrics = {"hints": [], "bad_discards": 0, "plays": []}

    def _build_deck(self, rng):
        deck = []
        for c in range(V2_COLORS):
            for rank, count in enumerate(V2_RANK_DIST):
                deck.extend([(c, rank)] * count)
        rng.shuffle(deck)
        return deck

    def _new_knowledge(self):
        return {"color": None, "rank": None}

    def reset(self, rng=None, **kwargs):
        rng = rng or np.random.default_rng()
        self.deck = self._build_deck(rng)
        self.hands = [[], []]
        self.knowledge = [[], []]
        self.fireworks = [0] * V2_COLORS
        self.discard = []
        self.hints = V2_INITIAL_HINTS
        self.lives = V2_LIVES
        self.step_count = 0
        self.current_player_id = 0
        self.done = False
        self.score = 0.0
        self.metrics = {"hints": [], "bad_discards": 0, "plays": []}
        for player in [0, 1]:
            for _ in range(V2_HAND_SIZE):
                self._draw(player)
        return self

    @property
    def current_player(self):
        return self.current_player_id

    @property
    def is_terminal(self):
        return self.done

    @property
    def returns(self):
        return [self.score, self.score]

    def _draw(self, player):
        if self.deck:
            self.hands[player].append(self.deck.pop())
            self.knowledge[player].append(self._new_knowledge())

    def _replace(self, player, idx):
        if 0 <= idx < len(self.hands[player]):
            self.hands[player].pop(idx)
            self.knowledge[player].pop(idx)
            self._draw(player)

    def legal_actions(self):
        if self.done:
            return []
        actions = []
        for slot in range(len(self.hands[self.current_player_id])):
            actions.append(V2_PLAY_OFFSET + slot)
            actions.append(V2_DISCARD_OFFSET + slot)
        if self.hints > 0:
            partner = 1 - self.current_player_id
            partner_colors = {c for c, _ in self.hands[partner]}
            partner_ranks = {r for _, r in self.hands[partner]}
            for c in partner_colors:
                actions.append(V2_HINT_COLOR_OFFSET + c)
            for r in partner_ranks:
                actions.append(V2_HINT_RANK_OFFSET + r)
        return actions if actions else [V2_DISCARD_OFFSET]

    def info_state(self, player):
        partner = 1 - player
        partner_cards = ",".join(f"{c}{r}" for c, r in self.hands[partner])
        own_knowledge = ";".join(
            f"{(k['color'] if k['color'] is not None else '_')}"
            f"{(k['rank'] if k['rank'] is not None else '_')}"
            for k in self.knowledge[player]
        )
        fireworks = "".join(str(x) for x in self.fireworks)
        return (
            f"p{player}|hand={own_knowledge}|partner={partner_cards}"
            f"|fw={fireworks}|h={self.hints}|l={self.lives}"
        )

    def step(self, action):
        assert not self.done
        player = self.current_player_id
        if V2_PLAY_OFFSET <= action < V2_PLAY_OFFSET + V2_HAND_SIZE:
            self._play(player, action - V2_PLAY_OFFSET)
        elif V2_DISCARD_OFFSET <= action < V2_DISCARD_OFFSET + V2_HAND_SIZE:
            self._discard(player, action - V2_DISCARD_OFFSET)
        elif V2_HINT_COLOR_OFFSET <= action < V2_HINT_COLOR_OFFSET + V2_COLORS:
            self._hint(player, "color", action - V2_HINT_COLOR_OFFSET)
        elif V2_HINT_RANK_OFFSET <= action < V2_HINT_RANK_OFFSET + V2_RANKS:
            self._hint(player, "rank", action - V2_HINT_RANK_OFFSET)
        else:
            raise ValueError(action)

        self.step_count += 1
        max_score = float(V2_COLORS * V2_RANKS)
        if self.lives <= 0 or self.score >= max_score or self.step_count >= self.max_steps:
            self.done = True
        self.current_player_id = 1 - self.current_player_id

    def _play(self, player, slot):
        if slot >= len(self.hands[player]):
            return
        card = self.hands[player][slot]
        color, rank = card
        self.metrics["plays"].append(card)
        if rank == self.fireworks[color]:
            self.fireworks[color] += 1
            self.score += 1.0
        else:
            self.lives -= 1
            self.discard.append(card)
        self._replace(player, slot)

    def _discard(self, player, slot):
        if slot >= len(self.hands[player]):
            return
        card = self.hands[player][slot]
        color, rank = card
        if rank == self.fireworks[color]:
            self.metrics["bad_discards"] += 1
        self.discard.append(card)
        self.hints = min(V2_MAX_HINTS, self.hints + 1)
        self._replace(player, slot)

    def _hint(self, player, kind, value):
        partner = 1 - player
        if self.hints <= 0:
            return
        matched = False
        for slot, (c, r) in enumerate(self.hands[partner]):
            if (kind == "color" and c == value) or (kind == "rank" and r == value):
                self.knowledge[partner][slot][kind] = value
                matched = True
        if matched:
            self.hints -= 1
            self.metrics["hints"].append((kind, value))


class MaskedHanabiV2:
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

    @property
    def is_terminal(self):
        return self.env.is_terminal

    @property
    def returns(self):
        return self.env.returns

    @property
    def metrics(self):
        return self.env.metrics

    def info_state(self, player):
        return self.env.info_state(player)

    def legal_actions(self):
        actions = self.env.legal_actions()
        if self.mask_fn and not self.env.is_terminal:
            player = self.env.current_player
            info = self.env.info_state(player)
            masked = self.mask_fn(info, actions, player)
            if masked:
                return masked
        return actions

    def step(self, action):
        self.env.step(action)

