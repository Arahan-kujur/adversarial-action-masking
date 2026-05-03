"""Leduc Poker environment with masked wrapper.

6 cards (J, Q, K x 2 suits), 3 actions, 2 betting rounds, fixed-limit.
"""

import numpy as np

FOLD, CHECK_CALL, RAISE = 0, 1, 2
NUM_ACTIONS = 3
CARD_RANKS = [0, 1, 2]  # J, Q, K
CARD_NAMES = {0: "J", 1: "Q", 2: "K"}
DECK = [(r, s) for r in CARD_RANKS for s in range(2)]  # 6 cards total


class LeducPokerEnv:
    """Leduc Poker: 6 cards, 2 players, 3 actions, 2 rounds, fixed-limit.

    Round 1 raise = 2 chips, round 2 raise = 4 chips.
    Max 2 raises per round.  Each player antes 1 chip.

    Card representation: each card is ``(rank, suit)`` where rank in {0,1,2}
    and suit in {0,1}.
    """

    def __init__(self):
        self._hands: list = [None, None]
        self._community = None
        self._round: int = 0          # 0 = pre-flop, 1 = post-flop
        self._history: list[list[int]] = [[], []]  # per-round action lists
        self._done: bool = False
        self._reward_p0: float = 0.0
        self._pot: list[int] = [1, 1]  # ante
        self._raises_this_round: int = 0
        self._round_starter: int = 0
        self._deck_remaining: list = []

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, cards=None, rng=None):
        """Reset the game.

        ``cards`` can be ``(hand0, hand1)`` or ``(hand0, hand1, community)``
        where each hand/card is a ``(rank, suit)`` tuple.
        """
        if cards is not None:
            self._hands = [cards[0], cards[1]]
            if len(cards) >= 3:
                self._community = cards[2]
                remaining = [c for c in DECK if c not in (cards[0], cards[1], cards[2])]
            else:
                self._community = None
                remaining = [c for c in DECK if c not in (cards[0], cards[1])]
            self._deck_remaining = remaining
        else:
            deck = list(DECK)
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

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_player(self) -> int:
        if self._done:
            return -1
        actions = self._history[self._round]
        if len(actions) == 0:
            return self._round_starter
        return (self._round_starter + len(actions)) % 2

    @property
    def is_terminal(self) -> bool:
        return self._done

    @property
    def returns(self) -> list[float]:
        return [self._reward_p0, -self._reward_p0]

    # ------------------------------------------------------------------
    # Info state
    # ------------------------------------------------------------------

    def info_state(self, player: int) -> str:
        """String encoding visible to ``player``.

        Format: ``"rank|community_rank_or_underscore|history_chars"``

        History chars: ``f`` = fold, ``c`` = check/call, ``r`` = raise.
        A ``/`` separates rounds.
        """
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

    # ------------------------------------------------------------------
    # Legal actions
    # ------------------------------------------------------------------

    def legal_actions(self) -> list[int]:
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

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action: int) -> None:
        """Apply *action* for the current player."""
        assert not self._done
        assert action in self.legal_actions(), (
            f"Illegal action {action}, legal: {self.legal_actions()}"
        )

        player = self.current_player
        self._history[self._round].append(action)

        raise_size = 2 if self._round == 0 else 4

        if action == FOLD:
            winner = 1 - player
            self._done = True
            self._reward_p0 = self._pot[0] if winner == 1 else self._pot[1]
            self._reward_p0 = float(self._pot[1 - 0] if winner == 0 else -self._pot[0])
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

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _round_over(self) -> bool:
        """Check whether the current betting round is complete."""
        actions = self._history[self._round]
        if len(actions) < 2:
            return False
        last_two = actions[-2:]
        if last_two == [CHECK_CALL, CHECK_CALL]:
            return True
        if last_two[-1] == CHECK_CALL and last_two[-2] == RAISE:
            return True
        return False

    def _advance_to_round_1(self) -> None:
        """Deal community card and start round 2."""
        if self._community is None:
            if self._deck_remaining:
                self._community = self._deck_remaining[0]
            else:
                self._community = (0, 0)
        self._round = 1
        self._raises_this_round = 0
        self._round_starter = 0

    def _showdown(self) -> None:
        """Resolve the hand at showdown."""
        self._done = True
        r0 = self._hand_rank(0)
        r1 = self._hand_rank(1)
        if r0 > r1:
            self._finalize(winner=0)
        elif r1 > r0:
            self._finalize(winner=1)
        else:
            self._reward_p0 = 0.0

    def _hand_rank(self, player: int) -> int:
        """Higher is better.  Pair beats high card."""
        card_rank = self._hands[player][0]
        if self._community is not None and card_rank == self._community[0]:
            return 100 + card_rank  # pair
        return card_rank  # high card

    def _finalize(self, winner: int) -> None:
        """Set reward from the winner's perspective."""
        self._done = True
        winnings = self._pot[1 - winner]
        self._reward_p0 = float(winnings) if winner == 0 else float(-self._pot[0])


# -----------------------------------------------------------------------
# Masked wrapper (mirrors MaskedKuhnPoker)
# -----------------------------------------------------------------------

class MaskedLeducPoker:
    """Wraps LeducPokerEnv with an action mask applied per-step."""

    def __init__(self, env: LeducPokerEnv):
        self.env = env
        self.mask_fn = None

    def set_mask(self, mask_fn):
        """Set mask function: ``mask_fn(info_state, legal_actions, player) -> filtered_actions``."""
        self.mask_fn = mask_fn

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        return self

    @property
    def current_player(self) -> int:
        return self.env.current_player

    def info_state(self, player: int) -> str:
        return self.env.info_state(player)

    def legal_actions(self) -> list[int]:
        actions = self.env.legal_actions()
        if self.mask_fn is not None and not self.env.is_terminal:
            player = self.env.current_player
            info = self.env.info_state(player)
            masked = self.mask_fn(info, actions, player)
            if masked:
                return masked
        return actions

    @property
    def is_terminal(self) -> bool:
        return self.env.is_terminal

    @property
    def returns(self) -> list[float]:
        return self.env.returns

    def step(self, action: int) -> None:
        self.env.step(action)
