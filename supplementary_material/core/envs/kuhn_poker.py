"""Kuhn Poker environment -- minimal, reusable."""

import numpy as np

PASS, BET = 0, 1
NUM_ACTIONS = 2
CARDS = [0, 1, 2]
CARD_NAMES = {0: "J", 1: "Q", 2: "K"}


class KuhnPokerEnv:
    """Kuhn Poker: 3 cards (J<Q<K), 2 players, 2 actions, ante 1."""

    def __init__(self):
        self._cards = [0, 0]
        self._history = []
        self._done = False
        self._reward_p0 = 0

    def reset(self, cards=None, rng=None):
        if cards is not None:
            self._cards = list(cards)
        else:
            deck = np.array(CARDS)
            rng.shuffle(deck)
            self._cards = [int(deck[0]), int(deck[1])]
        self._history = []
        self._done = False
        self._reward_p0 = 0
        return self

    @property
    def current_player(self):
        n = len(self._history)
        if n == 0: return 0
        if n == 1: return 1
        if n == 2: return 0
        return -1

    def info_state(self, player):
        card = self._cards[player]
        h = "".join("p" if a == PASS else "b" for a in self._history)
        return f"{card}{h}"

    def legal_actions(self):
        if self._done:
            return []
        return [PASS, BET]

    @property
    def is_terminal(self):
        return self._done

    @property
    def returns(self):
        return [self._reward_p0, -self._reward_p0]

    def step(self, action):
        assert not self._done
        self._history.append(action)
        h = tuple(self._history)
        showdown = 1 if self._cards[0] > self._cards[1] else -1
        payoffs = {
            (PASS, PASS): showdown,
            (BET, PASS): 1,
            (BET, BET): 2 * showdown,
            (PASS, BET, PASS): -1,
            (PASS, BET, BET): 2 * showdown,
        }
        if h in payoffs:
            self._done = True
            self._reward_p0 = payoffs[h]


class MaskedKuhnPoker:
    """Wraps KuhnPokerEnv with an action mask applied per-step."""

    def __init__(self, env):
        self.env = env
        self.mask_fn = None

    def set_mask(self, mask_fn):
        """Set mask function: mask_fn(info_state, legal_actions, player) -> filtered_actions."""
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
