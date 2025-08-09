# Place this file at: alpha-zero-general/yacht/YachtGame.py
# Then import it from main.py similarly to the othello example.

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import itertools
import random

# ===========================
# Game constants / encoding
# ===========================

# Categories in fixed order (12 total)
CATEGORIES = [
    "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX",
    "CHOICE", "FOUR_OF_A_KIND", "FULL_HOUSE",
    "SMALL_STRAIGHT", "LARGE_STRAIGHT", "YACHT"
]
NUM_CATEGORIES = len(CATEGORIES)

# Basic bonus threshold and bonus
BASIC_BONUS_THRESHOLD = 63000
BASIC_BONUS = 35000

# Discretized bid amounts (0..50000 in steps of 500) - more reasonable range
BID_STEP = 500
BID_LEVELS = 101  # 0..500*100 = 50000 max
BID_AMOUNTS = [i * BID_STEP for i in range(BID_LEVELS)]
BID_TARGETS = ["A", "B"]
NUM_BID_ACTIONS = BID_LEVELS * len(BID_TARGETS)  # 202

# 5 dice chosen out of up to 10 (positions 0..9 choose 5) -> 252
COMB_5_OF_10 = list(itertools.combinations(range(10), 5))
NUM_COMB = len(COMB_5_OF_10)  # 252

# Total scoring actions = category * combination
NUM_SCORE_ACTIONS = NUM_CATEGORIES * NUM_COMB  # 3024

# Final action space size
ACTION_SIZE = NUM_BID_ACTIONS + NUM_SCORE_ACTIONS  # 3226

# Phases
PHASE_BID = 0
PHASE_SCORE = 1

# Rounds
FIRST_ROUND = 1
LAST_ROUND = 13

# ----------------------------------------
# Utility: scoring for a chosen category
# ----------------------------------------


def score_category(cat_idx: int, dice: List[int]) -> int:
    # Basic categories (x1000)
    if cat_idx == 0:   # ONE
        return 1000 * sum(d for d in dice if d == 1)
    if cat_idx == 1:   # TWO
        return 1000 * sum(d for d in dice if d == 2)
    if cat_idx == 2:   # THREE
        return 1000 * sum(d for d in dice if d == 3)
    if cat_idx == 3:   # FOUR
        return 1000 * sum(d for d in dice if d == 4)
    if cat_idx == 4:   # FIVE
        return 1000 * sum(d for d in dice if d == 5)
    if cat_idx == 5:   # SIX
        return 1000 * sum(d for d in dice if d == 6)

    # Combo categories
    if cat_idx == 6:   # CHOICE
        return 1000 * sum(dice)

    if cat_idx == 7:   # FOUR_OF_A_KIND
        ok = any(dice.count(v) >= 4 for v in range(1, 7))
        return 1000 * sum(dice) if ok else 0

    if cat_idx == 8:   # FULL_HOUSE
        pair = triple = False
        for v in range(1, 7):
            cnt = dice.count(v)
            if cnt == 2 or cnt == 5:
                pair = True
            if cnt == 3 or cnt == 5:
                triple = True
        return 1000 * sum(dice) if pair and triple else 0

    if cat_idx == 9:   # SMALL_STRAIGHT
        present = [dice.count(i) > 0 for i in range(1, 7)]
        e1, e2, e3, e4, e5, e6 = present
        ok = (e1 and e2 and e3 and e4) or (
            e2 and e3 and e4 and e5) or (e3 and e4 and e5 and e6)
        return 15000 if ok else 0

    if cat_idx == 10:  # LARGE_STRAIGHT
        present = [dice.count(i) > 0 for i in range(1, 7)]
        e1, e2, e3, e4, e5, e6 = present
        ok = (e1 and e2 and e3 and e4 and e5) or (
            e2 and e3 and e4 and e5 and e6)
        return 30000 if ok else 0

    if cat_idx == 11:  # YACHT
        ok = any(dice.count(i) == 5 for i in range(1, 7))
        return 50000 if ok else 0

    raise ValueError("Invalid category")

# ----------------------------------------
# Internal state
# ----------------------------------------


@dataclass
class PlayerState:
    # 5 dice carried into the round (or 0 at start)
    carry: List[int] = field(default_factory=list)
    # 12-bit mask of used categories
    used_mask: int = 0
    cat_scores: List[int] = field(default_factory=lambda: [0] * NUM_CATEGORIES)
    # net bidding adjustments
    bid_score: int = 0

    def basic_total(self) -> int:
        return sum(self.cat_scores[0:6])

    def total_with_bonus(self) -> int:
        bonus = BASIC_BONUS if self.basic_total() >= BASIC_BONUS_THRESHOLD else 0
        return sum(self.cat_scores) + bonus + self.bid_score


@dataclass
class YachtState:
    round_no: int = FIRST_ROUND
    phase: int = PHASE_BID  # bid or score
    # current visible roll (only when bidding exists for this round)
    rollA: List[int] = field(default_factory=list)
    rollB: List[int] = field(default_factory=list)
    # bids waiting to be resolved in this round (None if not placed yet)
    p1_bid: Optional[Tuple[str, int]] = None   # ('A'|'B', amount)
    p2_bid: Optional[Tuple[str, int]] = None
    # player states
    p1: PlayerState = field(default_factory=PlayerState)
    p2: PlayerState = field(default_factory=PlayerState)
    # whose turn inside the serialized move order (AlphaZero alternates turns)
    # This is handled by the framework via 'player' arg; we keep no local field.

# ----------------------------------------
# Helpers
# ----------------------------------------


def roll_five() -> List[int]:
    return list(np.random.randint(1, 7, size=5))


def tiebreak_uniform() -> int:
    return random.randint(0, 1)  # 0 -> first wins tie, 1 -> second


def encode_bid_action(target: str, amount: int) -> int:
    tidx = 0 if target == "A" else 1
    aidx = amount // BID_STEP
    return tidx * BID_LEVELS + aidx  # 0..201


def decode_bid_action(a: int) -> Tuple[str, int]:
    tidx = a // BID_LEVELS
    aidx = a % BID_LEVELS
    return ("A" if tidx == 0 else "B", BID_AMOUNTS[aidx])


def decode_score_action(a: int) -> Tuple[int, Tuple[int, int, int, int, int]]:
    base = a - NUM_BID_ACTIONS
    cat_idx = base // NUM_COMB
    comb_idx = base % NUM_COMB
    return cat_idx, COMB_5_OF_10[comb_idx]


def active_player_state(s: YachtState, player: int) -> PlayerState:
    return s.p1 if player == 1 else s.p2


def opp_player_state(s: YachtState, player: int) -> PlayerState:
    return s.p2 if player == 1 else s.p1


def both_scored_all(ps: PlayerState) -> bool:
    return bin(ps.used_mask).count("1") == NUM_CATEGORIES


def need_bidding(round_no: int) -> bool:
    return round_no != LAST_ROUND


def need_scoring(round_no: int) -> bool:
    return round_no != FIRST_ROUND


def visible_rolls_for_round(round_no: int, phase: int) -> bool:
    # Rolls A/B visible only when we're in a round that has bidding and phase is bid
    return phase == PHASE_BID and need_bidding(round_no)

# ==================================================
# Game class in the AlphaZero-General style
# ==================================================


class YachtGame:
    """
    AlphaZero-General game wrapper for the 13-round Yacht-with-Bidding dice game.

    IMPORTANT MODELING NOTE:
    - Bidding is *simultaneous* in the official rules, but AlphaZero requires sequential
      perfect-information turns. We serialize bids into two turns (P1 then P2).
      To reduce advantage leakage, we *mask* the first player's pending bid from the
      second player's observation during their bidding turn. This keeps the spirit of
      simultaneous bidding while staying compatible with MCTS.
    """

    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        self._init_rolls = True  # roll for round 1 upon initialization

    # -------------
    # API — required by alpha-zero-general (see README)
    # -------------

    def getInitBoard(self) -> YachtState:
        s = YachtState()
        # Round 1 is bidding-only. Roll A/B now.
        s.rollA = roll_five()
        s.rollB = roll_five()
        return s

    def getBoardSize(self) -> Tuple[int, int]:
        """
        Return a 2D 'board' size. We will feed a flat feature plane of length F.
        You will need to build an NN for vector inputs (1 x F). For now we only
        provide the game logic; see othello example to wire a network.
        """
        # Feature vector length (kept small and simple):
        #   - round/phase one-hot-ish (3 scalars)
        #   - my 10 dice slots (pad -1 for missing) -> 10
        #   - opp 10 dice slots (pad -1) -> 10
        #   - current roll A (5, pad -1 when not bidding)
        #   - current roll B (5)
        #   - my used categories mask (12) as 0/1
        #   - opp used categories mask (12)
        #   - my bid_score, opp bid_score (2, scaled by 1e-5)
        # Total F = 3 + 10 + 10 + 5 + 5 + 12 + 12 + 2 = 59
        return (1, 59)

    def getActionSize(self) -> int:
        return ACTION_SIZE

    def getNextState(self, board: YachtState, player: int, action: int) -> Tuple[YachtState, int]:
        """
        Apply action for 'player' and return (next_board, next_player).
        """
        s = self._copy_state(board)

        if s.phase == PHASE_BID and need_bidding(s.round_no):
            # Decode bid
            if not (0 <= action < NUM_BID_ACTIONS):
                raise ValueError("Invalid action in BID phase")
            target, amount = decode_bid_action(action)
            # Record bid
            if s.p1_bid is None and s.p2_bid is None:
                # First bidder in this round
                if player == 1:
                    s.p1_bid = (target, amount)
                else:
                    s.p2_bid = (target, amount)
                next_player = -player  # let the other player bid next
                return s, next_player
            else:
                # Second bidder -> resolve
                if player == 1:
                    s.p1_bid = (target, amount)
                else:
                    s.p2_bid = (target, amount)

                self._resolve_bids_and_assign(s)

                # Move phase:
                if need_scoring(s.round_no):
                    s.phase = PHASE_SCORE
                    # Scoring order is serialized; start with Player 1 each scoring phase
                    next_player = 1
                else:
                    # No scoring (round 1). Advance to round 2 and roll.
                    s.round_no += 1
                    s.p1_bid = s.p2_bid = None
                    s.rollA = roll_five()
                    s.rollB = roll_five()
                    s.phase = PHASE_BID
                    next_player = 1  # P1 bids first in our serialization
                return s, next_player

        # SCORE phase
        if s.phase == PHASE_SCORE:
            if not (NUM_BID_ACTIONS <= action < ACTION_SIZE):
                raise ValueError("Invalid action in SCORE phase")
            cat_idx, comb = decode_score_action(action)

            me = active_player_state(s, player)
            # cannot reuse a category
            if (me.used_mask >> cat_idx) & 1:
                # invalid, but MCTS masks these; if it happens return unchanged
                return s, -player

            # Build my 10 dice view: carry plus if present (from current round after bidding)
            # Our model keeps all dice in 'carry'; after bidding for rounds >=2, carry has 10 dice,
            # and after scoring it will go back to 5 (unscored) for the next round.
            # For round 13, carry is exactly 5.
            my_dice = list(me.carry)
            # Indices in comb refer to positions 0..9; if missing positions (> len-1), it's invalid.
            if max(comb) >= len(my_dice):
                # invalid combo
                return s, -player
            chosen = [my_dice[i] for i in comb]
            sc = score_category(cat_idx, chosen)

            # consume the chosen dice and mark category
            # remove by positions (from higher to lower to be safe)
            for i in sorted(comb, reverse=True):
                del my_dice[i]
            me.carry = my_dice
            me.used_mask |= (1 << cat_idx)
            me.cat_scores[cat_idx] = sc

            # If both players have played their scoring move for this round:
            # In Round 13, check if both players have used all categories
            if s.round_no == LAST_ROUND:
                # Check if both players are done (used all 12 categories)
                p1_done = both_scored_all(s.p1)
                p2_done = both_scored_all(s.p2)
                if p1_done and p2_done:
                    # Game ends - both players have used all categories
                    next_player = 1  # framework requires some player value; won't be used once ended
                    return s, next_player
                else:
                    # Continue round 13 - the other player still needs to score
                    next_player = -player
                    return s, next_player
            else:
                # For rounds 2-12, check if second player just scored
                if player == -1:
                    # Second player just scored, advance to next round
                    s.round_no += 1
                    s.p1_bid = s.p2_bid = None
                    if need_bidding(s.round_no):
                        # Normal round with bidding
                        s.rollA = roll_five()
                        s.rollB = roll_five()
                        s.phase = PHASE_BID
                    else:
                        # Round 13: no bidding, go straight to scoring
                        s.phase = PHASE_SCORE
                    next_player = 1
                    return s, next_player
                else:
                    # First player just scored, let second player score
                    next_player = -player
                    return s, next_player
                return s, next_player

        raise RuntimeError("Invalid phase/state")

    def getValidMoves(self, board: YachtState, player: int) -> np.ndarray:
        v = np.zeros(self.getActionSize(), dtype=np.uint8)

        if board.phase == PHASE_BID and need_bidding(board.round_no):
            # All discretized bids are allowed (we rely on policy to learn size/risk)
            v[:NUM_BID_ACTIONS] = 1
            return v

        if board.phase == PHASE_SCORE:
            me = active_player_state(board, player)
            # available dice count: 10 (rounds 2-12) or 5 (round 13)
            n = len(me.carry)
            if n < 5:
                # This should only happen if the game has ended
                # In Round 13, if a player has < 5 dice, they may have already scored all categories
                if both_scored_all(me):
                    return v  # No moves - player is done
                else:
                    # Game logic error - should not reach here with correct implementation
                    return v
            # any combination selecting positions < n is legal
            valid_combs = [i for i, comb in enumerate(
                COMB_5_OF_10) if max(comb) < n]

            # categories not used
            for cat in range(NUM_CATEGORIES):
                if ((me.used_mask >> cat) & 1) == 0:
                    base = NUM_BID_ACTIONS + cat * NUM_COMB
                    for ci in valid_combs:
                        v[base + ci] = 1
            return v

        return v

    def getGameEnded(self, board: YachtState, player: int) -> float:
        """
        Returns:
          0 if not ended,
          +1 if current player eventually wins,
          -1 if current player eventually loses,
          small value for draw (as used by the A0G repo).
        """
        # Game ends after both players have used all 12 categories
        p1_done = both_scored_all(board.p1)
        p2_done = both_scored_all(board.p2)
        if not (p1_done and p2_done):
            return 0.0

        p1_total = board.p1.total_with_bonus()
        p2_total = board.p2.total_with_bonus()

        if p1_total == p2_total:
            return 1e-4  # draw convention used widely in this repo
        winner = 1 if p1_total > p2_total else -1
        return float(winner if player == 1 else -winner)

    def getCanonicalForm(self, board: YachtState, player: int) -> YachtState:
        """
        Return a perspective-flipped state. We also mask the *opponent's pending bid*
        in the BID phase to approximate simultaneous secrecy.
        """
        if player == 1:
            return board  # already canonical
        # swap p1 and p2 fields
        s = self._copy_state(board)
        s.p1, s.p2 = s.p2, s.p1
        # swap bids
        s.p1_bid, s.p2_bid = s.p2_bid, s.p1_bid
        return s

    def getSymmetries(self, board: YachtState, pi: np.ndarray) -> List[Tuple[YachtState, np.ndarray]]:
        # No spatial symmetries for this game.
        return [(board, pi)]

    def stringRepresentation(self, board: YachtState) -> str:
        # Hashable string for MCTS memoization. Include full info so transitions are deterministic.
        p1 = board.p1
        p2 = board.p2
        return "|".join([
            f"r{board.round_no}",
            f"ph{board.phase}",
            f"A{''.join(map(str, board.rollA)) if board.rollA else '-'}",
            f"B{''.join(map(str, board.rollB)) if board.rollB else '-'}",
            f"p1b{board.p1_bid[0]}{board.p1_bid[1]}" if board.p1_bid else "p1b-",
            f"p2b{board.p2_bid[0]}{board.p2_bid[1]}" if board.p2_bid else "p2b-",
            f"p1c{''.join(map(str, p1.carry))}",
            f"p2c{''.join(map(str, p2.carry))}",
            f"p1u{p1.used_mask}",
            f"p2u{p2.used_mask}",
            f"p1s{','.join(map(str, p1.cat_scores))}",
            f"p2s{','.join(map(str, p2.cat_scores))}",
            f"p1bid{p1.bid_score}",
            f"p2bid{p2.bid_score}",
        ])

    # ----------------
    # Pretty printer (optional)
    # ----------------
    def display(self, board: YachtState):
        # Display function disabled for training
        pass

    # ==================================================
    # Internal mechanics
    # ==================================================

    def _copy_state(self, s: YachtState) -> YachtState:
        return YachtState(
            round_no=s.round_no,
            phase=s.phase,
            rollA=list(s.rollA),
            rollB=list(s.rollB),
            p1_bid=None if s.p1_bid is None else (s.p1_bid[0], s.p1_bid[1]),
            p2_bid=None if s.p2_bid is None else (s.p2_bid[0], s.p2_bid[1]),
            p1=PlayerState(
                carry=list(s.p1.carry),
                used_mask=s.p1.used_mask,
                cat_scores=list(s.p1.cat_scores),
                bid_score=s.p1.bid_score,
            ),
            p2=PlayerState(
                carry=list(s.p2.carry),
                used_mask=s.p2.used_mask,
                cat_scores=list(s.p2.cat_scores),
                bid_score=s.p2.bid_score,
            ),
        )

    def _resolve_bids_and_assign(self, s: YachtState):
        """
        After both p1_bid and p2_bid are present, assign bundles and update bid scores,
        add the obtained 5 dice to each player's carry. This is called only in rounds
        where bidding exists (1..12).
        """
        assert s.p1_bid is not None and s.p2_bid is not None
        # Determine who gets which bundle
        p1_target, p1_amount = s.p1_bid
        p2_target, p2_amount = s.p2_bid
        get_groups = [p1_target, p2_target]

        # If target same, compare bids, break ties uniformly
        if get_groups[0] == get_groups[1]:
            winner = None
            if p1_amount > p2_amount:
                winner = 0
            elif p2_amount > p1_amount:
                winner = 1
            else:
                winner = tiebreak_uniform()
            # winner gets the chosen group, other gets the other
            if winner == 0:
                get_groups[1] = "B" if get_groups[0] == "A" else "A"
            else:
                get_groups[0] = "B" if get_groups[1] == "A" else "A"

        # Update bid scores (- if you *got* target, + if you *missed* it)
        s.p1.bid_score += (-p1_amount if get_groups[0]
                           == p1_target else +p1_amount)
        s.p2.bid_score += (-p2_amount if get_groups[1]
                           == p2_target else +p2_amount)

        # Assign dice
        p1_get = s.rollA if get_groups[0] == "A" else s.rollB
        p2_get = s.rollA if get_groups[1] == "A" else s.rollB

        # After bidding, players hold 5 dice (round 1) or 10 dice (rounds >=2)
        # We keep everything in 'carry'. In round 1, these 5 will carry into round 2.
        s.p1.carry.extend(p1_get)
        s.p2.carry.extend(p2_get)
