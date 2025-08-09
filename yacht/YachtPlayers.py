# alpha-zero-general/yacht/YachtPlayers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import math
import random

# We import constants/helpers from the Game so scoring & indexing match exactly.
from yacht.YachtGame import (
    YachtState, PlayerState,
    PHASE_BID, PHASE_SCORE, LAST_ROUND,
    NUM_BID_ACTIONS, NUM_SCORE_ACTIONS, ACTION_SIZE,
    BID_STEP, BID_LEVELS, BID_TARGETS, BID_AMOUNTS,
    COMB_5_OF_10, NUM_COMB,
    CATEGORIES, NUM_CATEGORIES,
    BASIC_BONUS, BASIC_BONUS_THRESHOLD,
    score_category,
)

# ---------- small utilities (mirror the Game’s encoding) ----------


def _encode_bid_action(target: str, amount: int) -> int:
    tidx = 0 if target == "A" else 1
    aidx = amount // BID_STEP
    return tidx * BID_LEVELS + aidx


def _decode_score_action(a: int) -> Tuple[int, Tuple[int, int, int, int, int]]:
    base = a - NUM_BID_ACTIONS
    cat_idx = base // NUM_COMB
    comb_idx = base % NUM_COMB
    return cat_idx, COMB_5_OF_10[comb_idx]

# ---------- bidding heuristics ----------


def _score_potential_after_bid(state: YachtState, me: PlayerState, bundle: List[int]) -> int:
    """
    Estimate how valuable it is for ME to receive `bundle` *this round*.
    - If this round has scoring (round 2..12), we approximate by: best possible
      score I can put *now* after adding `bundle` to my carry (which gives 10 dice).
    - If round 1 (no scoring), use a forward-looking heuristic that rewards duplicates,
      straights, and raw sum.
    Returns an integer on the same scale as category scores (i.e., thousands).
    """
    # Round 1: forward heuristic
    if state.round_no == 1:
        dice = me.carry + bundle  # becomes my 5 dice carried to round 2
        # reward: CHOICE sum, plus small extras for structure
        s = 1000 * sum(dice)
        counts = [dice.count(v) for v in range(1, 7)]
        # encourage 3/4/5 of a kind and near straights
        if max(counts) >= 4:
            s += 6000
        elif max(counts) == 3:
            s += 3000
        # encourage runs 1234/2345/3456
        present = [c > 0 for c in counts]  # 1..6
        e1, e2, e3, e4, e5, e6 = present
        if (e1 and e2 and e3 and e4) or (e2 and e3 and e4 and e5) or (e3 and e4 and e5 and e6):
            s += 5000
        return s

    # Rounds 2..12: greedy best scoring you could play this phase after adding bundle
    my_dice = me.carry + bundle  # should be 10 dice available
    n = len(my_dice)
    if n < 5:
        return 0
    valid_combs = [comb for comb in COMB_5_OF_10 if max(comb) < n]

    # Compute current basic subtotal and used mask
    used = me.used_mask
    basic_before = sum(me.cat_scores[0:6])

    best = -10**18
    for cat in range(NUM_CATEGORIES):
        if ((used >> cat) & 1) != 0:
            continue
        for comb in valid_combs:
            chosen = [my_dice[i] for i in comb]
            sc = score_category(cat, chosen)
            # add bonus if this play crosses the basic threshold
            gain = sc
            if cat < 6:
                new_basic = basic_before + sc
                if basic_before < BASIC_BONUS_THRESHOLD <= new_basic:
                    gain += BASIC_BONUS
            if gain > best:
                best = gain
    return best if best != -10**18 else 0


def _choose_bid(state: YachtState) -> int:
    """Pick (target, amount) -> action index using a value-gap heuristic."""
    me = state.p1  # canonical perspective
    # Evaluate both bundles (if bidding exists this round)
    valA = _score_potential_after_bid(state, me, state.rollA)
    valB = _score_potential_after_bid(state, me, state.rollB)

    # target = the better bundle
    if valA >= valB:
        target = "A"
        gap = max(0, valA - valB)
    else:
        target = "B"
        gap = max(0, valB - valA)

    # scale bid by (value gap) and (current bid-score lead/deficit)
    # this is intentionally conservative; snap to 0..100000 in steps of 1000
    # The gap is on the same scale as points (thousands), so we can damp it.
    # Also bias by current score diff: if I'm behind (negative), bid a bit more.
    my_total = me.bid_score + \
        sum(me.cat_scores) + \
        (BASIC_BONUS if sum(me.cat_scores[0:6])
         >= BASIC_BONUS_THRESHOLD else 0)
    opp = state.p2
    opp_total = opp.bid_score + sum(opp.cat_scores) + (
        BASIC_BONUS if sum(opp.cat_scores[0:6]) >= BASIC_BONUS_THRESHOLD else 0)
    diff = my_total - opp_total

    # heuristic bid in thousands:
    bid_k = 0.5 * (gap / 1000.0) - 0.15 * (diff / 1000.0)
    # squashing and clipping
    bid = int(max(0, min(100000, round(1000 * bid_k))))
    # snap to legal grid (multiple of 1000)
    bid = (bid // BID_STEP) * BID_STEP
    return _encode_bid_action(target, bid)

# ---------- scoring heuristics ----------


def _choose_scoring(state: YachtState) -> int:
    """Greedy best-immediate-score (with basic-bonus awareness)."""
    me = state.p1  # canonical
    n = len(me.carry)
    if n < 5:
        # no-op fallback (shouldn't happen), return any masked action == 0
        return 0

    basic_before = sum(me.cat_scores[0:6])
    used = me.used_mask

    best_gain = -10**18
    best_action = None

    # enumerate all unused categories and all valid 5-index subsets
    for cat in range(NUM_CATEGORIES):
        if ((used >> cat) & 1) != 0:
            continue
        # valid combos are those that index only into 0..n-1
        base = NUM_BID_ACTIONS + cat * NUM_COMB
        for ci, comb in enumerate(COMB_5_OF_10):
            if max(comb) >= n:
                break  # later combos will also be invalid since list is sorted
            chosen = [me.carry[i] for i in comb]
            sc = score_category(cat, chosen)
            gain = sc
            if cat < 6:
                new_basic = basic_before + sc
                if basic_before < BASIC_BONUS_THRESHOLD <= new_basic:
                    gain += BASIC_BONUS
            if gain > best_gain:
                best_gain = gain
                best_action = base + ci

    # Fallback: if something went wrong
    return best_action if best_action is not None else 0

# ---------- Player classes ----------


class RandomYachtPlayer:
    """Uniform random legal move."""

    def __init__(self, game):
        self.game = game

    def play(self, board: YachtState) -> int:
        valids = self.game.getValidMoves(board, 1)  # canonical player = 1
        legal = np.nonzero(valids)[0]
        return int(np.random.choice(legal)) if len(legal) else 0


class GreedyYachtPlayer:
    """
    Heuristic bidder + greedy scorer.
    Works with the canonical board passed by Arena (as in the framework).
    """

    def __init__(self, game, seed: Optional[int] = None):
        self.game = game
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def play(self, board: YachtState) -> int:
        valids = self.game.getValidMoves(board, 1)
        if board.phase == PHASE_BID and board.round_no != LAST_ROUND:
            a = _choose_bid(board)
            if valids[a] == 1:
                return a
            # if masked out for any reason, fallback to random legal
            legal = np.nonzero(valids)[0]
            return int(np.random.choice(legal)) if len(legal) else 0

        # SCORE phase
        a = _choose_scoring(board)
        if valids[a] == 1:
            return a
        # fallback to random legal if our choice was somehow invalid
        legal = np.nonzero(valids)[0]
        return int(np.random.choice(legal)) if len(legal) else 0


class HumanYachtPlayer:
    """
    Simple CLI player — prints a hint (greedy choice) and asks for an action index.
    Useful for quick sanity checks with `pit.py`. Not used during training.
    """

    def __init__(self, game):
        self.game = game

    def play(self, board: YachtState) -> int:
        valids = self.game.getValidMoves(board, 1)
        hint = None
        if board.phase == PHASE_BID and board.round_no != LAST_ROUND:
            hint = _choose_bid(board)
            print(
                f"[HUMAN] BID phase. RollA={board.rollA} RollB={board.rollB}")
            print(f"[HUMAN] Suggested bid action: {hint}")
        else:
            hint = _choose_scoring(board)
            print(
                f"[HUMAN] SCORE phase. Carry={board.p1.carry} UsedMask={bin(board.p1.used_mask)}")
            print(f"[HUMAN] Suggested score action: {hint}")

        print(
            f"[HUMAN] Valid actions: {np.where(valids==1)[0][:20]} ... (total {int(valids.sum())})")
        while True:
            try:
                a = int(input("[HUMAN] Enter action index: ").strip())
                if 0 <= a < ACTION_SIZE and valids[a] == 1:
                    return a
                print("Invalid action (either out of range or illegal). Try again.")
            except Exception:
                print("Please input an integer action index.")
