"""Extended Leduc Poker (Leduc-N): parameterised by number of ranks.

Leduc-5: 5 ranks x 2 suits = 10 cards, 3 actions, 2 rounds.
~300+ P0 info states. Requires function approximation.
"""

import numpy as np

FOLD, CHECK_CALL, RAISE = 0, 1, 2
NUM_ACTIONS = 3


class LeducNPokerEnv:
    """Leduc Poker with N ranks (default 5).

    N ranks x 2 suits = 2N cards. Round 1 raise = 2, round 2 raise = 4.
    Max 2 raises per round. Each player antes 1 chip.
    """

    def __init__(self, num_ranks=5):
        self.num_ranks = num_ranks
        self.deck = [(r, s) for r in range(num_ranks) for s in range(2)]
        self._hands = [None, None]
        self._community = None
        self._round = 0
        self._history = [[], []]
        self._done = False
        self._reward_p0 = 0.0
        self._pot = [1, 1]
        self._raises_this_round = 0
        self._round_starter = 0
        self._deck_remaining = []

    def reset(self, cards=None, rng=None):
        if cards is not None:
            self._hands = [cards[0], cards[1]]
            if len(cards) >= 3:
                self._community = cards[2]
                remaining = [c for c in self.deck if c not in cards[:3]]
            else:
                self._community = None
                remaining = [c for c in self.deck if c not in cards[:2]]
            self._deck_remaining = remaining
        else:
            deck = list(self.deck)
            rng.shuffle(deck)
            self._hands = [deck[0], deck[1]]
            self._deck_remaining = deck[2:]
            self._community = None

        self._round = 0
        self._history = [[], []]
        self._done = False
        self._reward_p0 = 0.0
        self._pot = [1, 1]
        self._raises_this_round = 0
        self._round_starter = 0
        return self

    @property
    def current_player(self):
        if self._done:
            return -1
        actions = self._history[self._round]
        if len(actions) == 0:
            return self._round_starter
        return (self._round_starter + len(actions)) % 2

    @property
    def is_terminal(self):
        return self._done

    @property
    def returns(self):
        return [self._reward_p0, -self._reward_p0]

    def info_state(self, player):
        rank = self._hands[player][0]
        if self._round >= 1 and self._community is not None:
            comm = str(self._community[0])
        else:
            comm = "_"
        action_map = {FOLD: "f", CHECK_CALL: "c", RAISE: "r"}
        h0 = "".join(action_map[a] for a in self._history[0])
        h1 = "".join(action_map[a] for a in self._history[1])
        history = h0 if self._round == 0 else h0 + "/" + h1
        return f"{rank}|{comm}|{history}"

    def legal_actions(self):
        if self._done:
            return []
        actions = [CHECK_CALL]
        if self._raises_this_round < 2:
            actions.append(RAISE)
        if len(self._history[self._round]) > 0:
            last = self._history[self._round][-1]
            if last == RAISE:
                actions = [FOLD, CHECK_CALL]
                if self._raises_this_round < 2:
                    actions.append(RAISE)
        return sorted(actions)

    def step(self, action):
        assert not self._done
        assert action in self.legal_actions()
        player = self.current_player
        self._history[self._round].append(action)
        raise_size = 2 if self._round == 0 else 4

        if action == FOLD:
            winner = 1 - player
            self._done = True
            self._finalize(winner)
            return

        if action == RAISE:
            self._raises_this_round += 1
            self._pot[player] += raise_size

        if action == CHECK_CALL:
            deficit = max(self._pot[1 - player] - self._pot[player], 0)
            self._pot[player] += deficit

        if self._round_over():
            if self._round == 0:
                self._advance_to_round_1()
            else:
                self._showdown()

    def _round_over(self):
        actions = self._history[self._round]
        if len(actions) < 2:
            return False
        if actions[-2:] == [CHECK_CALL, CHECK_CALL]:
            return True
        if actions[-1] == CHECK_CALL and actions[-2] == RAISE:
            return True
        return False

    def _advance_to_round_1(self):
        if self._community is None:
            if self._deck_remaining:
                self._community = self._deck_remaining[0]
            else:
                self._community = (0, 0)
        self._round = 1
        self._raises_this_round = 0
        self._round_starter = 0

    def _showdown(self):
        self._done = True
        r0 = self._hand_rank(0)
        r1 = self._hand_rank(1)
        if r0 > r1:
            self._finalize(0)
        elif r1 > r0:
            self._finalize(1)
        else:
            self._reward_p0 = 0.0

    def _hand_rank(self, player):
        card_rank = self._hands[player][0]
        if self._community is not None and card_rank == self._community[0]:
            return 100 + card_rank
        return card_rank

    def _finalize(self, winner):
        self._done = True
        winnings = self._pot[1 - winner]
        self._reward_p0 = float(winnings) if winner == 0 else float(-self._pot[0])


class MaskedLeducNPoker:
    """Wraps LeducNPokerEnv with an action mask."""

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
