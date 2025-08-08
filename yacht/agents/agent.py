"""
A heuristic-based agent for the yacht dice bidding and scoring game.

This agent follows the same I/O protocol as described in `Instruction_eng.md` and
implements significantly improved logic over the provided `sample-code.py`.  It
uses a greedy evaluation of both the bidding and scoring phases:

* **Bidding phase** – When two 5‑die bundles are offered, the agent estimates
  the immediate scoring potential of each bundle by combining it with the
  current hand (the carried‑over dice from previous rounds) and greedily
  selecting the best possible 5‑dice subset/category combination for the
  upcoming scoring phase.  The difference between these evaluations drives
  both the choice of bundle (`A` or `B`) and the bid amount.  Higher
  differences lead to larger bids, but never exceeding the value the agent
  expects to gain from the preferred bundle.  This approach attempts to
  secure high‑value bundles without overspending on the bid.

* **Scoring phase** – Given up to 10 dice (5 carried over plus 5 newly won),
  the agent exhaustively considers all 5‑dice subsets and all unused
  categories.  It computes the immediate score of each candidate and picks
  the combination that yields the highest score.  In the event of ties it
  prefers the combination that leaves behind dice with the highest average
  pips, as those dice may be more useful in future rounds.  This greedy
  selection aims to accumulate points quickly while still retaining decent
  dice for later use.

While this agent does not employ machine learning or AlphaZero training,
it serves as a strong baseline that you can further refine or use as a
starting point for reinforcement‑learning approaches.  See the
accompanying `report.md` for details on how to integrate this game into
the AlphaZero‑general framework and train a neural network–based agent.
"""

from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple


class DiceRule(Enum):
    ONE = 0
    TWO = 1
    THREE = 2
    FOUR = 3
    FIVE = 4
    SIX = 5
    CHOICE = 6
    FOUR_OF_A_KIND = 7
    FULL_HOUSE = 8
    SMALL_STRAIGHT = 9
    LARGE_STRAIGHT = 10
    YACHT = 11


@dataclass
class Bid:
    group: str  # 'A' or 'B'
    amount: int  # 0–100000 inclusive


@dataclass
class DicePut:
    rule: DiceRule
    dice: List[int]


class GameState:
    """
    Internal representation of a player's state across the game.

    This class holds the dice currently in hand, the scores already
    recorded for each category, and the cumulative bidding adjustments.
    It provides helper methods for computing scores and tracking used
    categories.
    """

    def __init__(self) -> None:
        # Dice currently held (after bidding).  At most 5 dice in round 1,
        # 5 after each scoring phase, and 10 before scoring.
        self.dice: List[int] = []
        # Scores for each category; None if unused.
        self.rule_score: List[Optional[int]] = [None] * 12
        # Net score from bidding (negative if we paid, positive if opponent
        # overbid and we gained points).
        self.bid_score: int = 0

    def get_total_score(self) -> int:
        """Compute the player's current total score including bonuses."""
        basic = sum(score for score in self.rule_score[:6] if score is not None)
        bonus = 35000 if basic >= 63000 else 0
        combination = sum(score for score in self.rule_score[6:] if score is not None)
        return basic + bonus + combination + self.bid_score

    def bid(self, is_successful: bool, amount: int) -> None:
        """Apply the result of a bid: subtract if we won our target, add if we lost."""
        if is_successful:
            self.bid_score -= amount
        else:
            self.bid_score += amount

    def add_dice(self, new_dice: List[int]) -> None:
        """Add new dice obtained via bidding."""
        self.dice.extend(new_dice)

    def use_dice(self, put: DicePut) -> None:
        """Consume the selected dice and record the score for the chosen category."""
        if self.rule_score[put.rule.value] is not None:
            raise RuntimeError(f"Category {put.rule.name} already used")
        # Remove the selected dice from the player's hand
        for d in put.dice:
            try:
                self.dice.remove(d)
            except ValueError:
                raise RuntimeError(f"Tried to remove die {d} which is not in hand")
        # Compute and record the score
        self.rule_score[put.rule.value] = self.calculate_score(put)

    @staticmethod
    def calculate_score(put: DicePut) -> int:
        """Compute the score for a given category/dice combination."""
        rule, dice = put.rule, put.dice
        # Basic categories (sum of faces matching the category times 1000)
        if rule == DiceRule.ONE:
            return sum(d for d in dice if d == 1) * 1000
        if rule == DiceRule.TWO:
            return sum(d for d in dice if d == 2) * 1000
        if rule == DiceRule.THREE:
            return sum(d for d in dice if d == 3) * 1000
        if rule == DiceRule.FOUR:
            return sum(d for d in dice if d == 4) * 1000
        if rule == DiceRule.FIVE:
            return sum(d for d in dice if d == 5) * 1000
        if rule == DiceRule.SIX:
            return sum(d for d in dice if d == 6) * 1000
        # Choice: sum of all dice times 1000
        if rule == DiceRule.CHOICE:
            return sum(dice) * 1000
        # Four of a kind: at least four dice the same; otherwise 0
        if rule == DiceRule.FOUR_OF_A_KIND:
            ok = any(dice.count(i) >= 4 for i in range(1, 7))
            return sum(dice) * 1000 if ok else 0
        # Full house: a 3-of-a-kind and a pair (five of a kind counts as both)
        if rule == DiceRule.FULL_HOUSE:
            pair = triple = False
            for i in range(1, 7):
                cnt = dice.count(i)
                if cnt == 2 or cnt == 5:
                    pair = True
                if cnt == 3 or cnt == 5:
                    triple = True
            return sum(dice) * 1000 if pair and triple else 0
        # Small straight: any sequence of four consecutive numbers (1-2-3-4,
        # 2-3-4-5, 3-4-5-6)
        if rule == DiceRule.SMALL_STRAIGHT:
            e = [dice.count(i) > 0 for i in range(1, 7)]
            ok = (e[0] and e[1] and e[2] and e[3]) or (e[1] and e[2] and e[3] and e[4]) or (e[2] and e[3] and e[4] and e[5])
            return 15000 if ok else 0
        # Large straight: sequence of five consecutive numbers (1-2-3-4-5 or 2-3-4-5-6)
        if rule == DiceRule.LARGE_STRAIGHT:
            e = [dice.count(i) > 0 for i in range(1, 7)]
            ok = (e[0] and e[1] and e[2] and e[3] and e[4]) or (e[1] and e[2] and e[3] and e[4] and e[5])
            return 30000 if ok else 0
        # Yacht: five of a kind
        if rule == DiceRule.YACHT:
            ok = any(dice.count(i) == 5 for i in range(1, 7))
            return 50000 if ok else 0
        raise RuntimeError(f"Unknown rule {rule}")


class Game:
    """
    Wrapper class managing both players' states and implementing bidding and scoring logic.

    This class maintains internal GameState objects for the agent and the opponent
    and exposes high‑level methods for deciding bids and scoring decisions.
    """

    def __init__(self) -> None:
        self.my_state = GameState()
        self.opp_state = GameState()
        # Track round number (1–13) to adjust strategy if desired
        self.round = 1

    # ---------------------------- Bidding Logic -----------------------------
    def calculate_bid(self, dice_a: List[int], dice_b: List[int]) -> Bid:
        """
        Decide which bundle to target and how much to bid.

        The agent evaluates each bundle by simulating the best possible scoring
        outcome if that bundle were added to its current dice pool.  The
        difference between the two evaluations influences both the chosen group
        (`A` vs `B`) and the bid amount.  The bid is capped to the expected
        advantage of the preferred bundle to avoid overpaying.
        """
        # Evaluate both bundles by estimating the maximum score achievable in
        # the next scoring phase if we were to take the bundle.
        val_a, _ = self._evaluate_bundle(dice_a)
        val_b, _ = self._evaluate_bundle(dice_b)
        # Determine preferred group
        if val_a > val_b:
            preferred_group = "A"
            diff = val_a - val_b
        elif val_b > val_a:
            preferred_group = "B"
            diff = val_b - val_a
        else:
            # If both bundles appear equal under our evaluation, break ties by
            # choosing the one with higher raw sum as a weak heuristic.
            if sum(dice_a) > sum(dice_b):
                preferred_group = "A"
            elif sum(dice_b) > sum(dice_a):
                preferred_group = "B"
            else:
                # If still tied, default to A
                preferred_group = "A"
            diff = 0

        # Convert the difference in potential score (points) into an
        # appropriate bid.  Because scores are in multiples of 1000, we scale
        # down the bid by 1000 to keep bids in a similar magnitude.  We also
        # avoid negative bids.
        # For example, if the expected advantage is 20000 points, the
        # resulting bid will be roughly 10 (half of 20) because we divide by
        # 1000 and scale by 0.5.  You can adjust the scaling factor for more
        # aggressive or conservative bidding.
        scaling_factor = 0.5
        bid_amount = int(diff / 1000 * scaling_factor)
        bid_amount = max(0, min(100000, bid_amount))

        # Incorporate current score difference into the bid.  If we're behind
        # significantly, we may want to bid more aggressively to catch up; if
        # we're ahead, a conservative approach is safer.  A simple linear
        # adjustment is applied here.
        my_score = self.my_state.get_total_score()
        opp_score = self.opp_state.get_total_score()
        score_diff = my_score - opp_score
        # If trailing, increase bid by 10% of the deficit scaled down
        if score_diff < 0:
            extra = int((-score_diff) / 10000)
            bid_amount += extra
        # If leading, decrease bid slightly
        elif score_diff > 0:
            bid_amount -= int(score_diff / 10000)

        # Clamp to valid range
        bid_amount = max(0, min(100000, bid_amount))
        return Bid(preferred_group, bid_amount)

    def _evaluate_bundle(self, bundle: List[int]) -> Tuple[int, Optional[Tuple[DiceRule, List[int]]]]:
        """
        Evaluate the immediate scoring potential of a bundle when combined with
        the current dice in hand.

        Returns a tuple `(max_score, best_choice)` where `max_score` is the
        highest possible score obtainable in the next scoring phase, and
        `best_choice` is the corresponding category and dice selection.  If
        no categories remain (should not happen before round 13), it returns
        `(0, None)`.
        """
        # Combine existing dice with the proposed bundle.  We simulate adding
        # the bundle to our current dice pool without mutating state.
        pool = self.my_state.dice + list(bundle)
        # Determine available categories
        available_rules = [DiceRule(i) for i, s in enumerate(self.my_state.rule_score) if s is None]
        if not available_rules:
            return 0, None
        max_score = -1
        best_choice = None  # Tuple[DiceRule, List[int]]
        # Generate all 5‑dice subsets of the pool
        # To reduce computation, if the pool has fewer than 5 dice (should
        # only happen in round 1), just take all of them.
        if len(pool) <= 5:
            subsets = [pool]
        else:
            # Use combinations from itertools to enumerate unique index sets
            subsets = []
            # Preindexing speeds up repeated counting; we generate combinations
            # of indices rather than values directly.
            for idxs in itertools.combinations(range(len(pool)), 5):
                subset = [pool[i] for i in idxs]
                subsets.append(subset)
        # Evaluate each subset for each available category
        for subset in subsets:
            for rule in available_rules:
                put = DicePut(rule, subset)
                score = GameState.calculate_score(put)
                if score > max_score:
                    max_score = score
                    best_choice = (rule, subset)
        return max_score, best_choice

    # ---------------------------- Scoring Logic -----------------------------
    def calculate_put(self) -> DicePut:
        """
        Decide which five dice to score and in which category.

        The agent examines every possible 5‑dice subset from its hand and
        every unused category, computing the resulting score for each.  It
        selects the combination with the highest immediate score.  If
        multiple candidates achieve the same score, the agent prefers the
        one leaving behind dice with the highest average pip value, as this
        heuristic tends to preserve useful dice for future categories.
        """
        # Determine which categories are still unused
        available_rules = [DiceRule(i) for i, s in enumerate(self.my_state.rule_score) if s is None]
        if not available_rules:
            # Should never happen before the final round
            raise RuntimeError("No available categories to score")
        # If fewer than 5 dice are present (only possible in the very last round),
        # we have no choice but to use them all.
        if len(self.my_state.dice) <= 5:
            pool_subsets = [list(self.my_state.dice)]
        else:
            pool_subsets = []
            for idxs in itertools.combinations(range(len(self.my_state.dice)), 5):
                pool_subsets.append([self.my_state.dice[i] for i in idxs])
        best_score = -1
        best_rule: Optional[DiceRule] = None
        best_subset: Optional[List[int]] = None
        best_leftover_avg = -1.0
        for subset in pool_subsets:
            # Compute leftover dice for tie‑breaking
            leftover = self._compute_leftover(self.my_state.dice, subset)
            leftover_avg = sum(leftover) / len(leftover) if leftover else 0.0
            for rule in available_rules:
                score = GameState.calculate_score(DicePut(rule, subset))
                if score > best_score or (score == best_score and leftover_avg > best_leftover_avg):
                    best_score = score
                    best_rule = rule
                    best_subset = subset
                    best_leftover_avg = leftover_avg
        if best_rule is None or best_subset is None:
            # Fallback (should not occur): choose first available rule and first
            # subset.
            best_rule = available_rules[0]
            best_subset = pool_subsets[0]
        return DicePut(best_rule, best_subset)

    @staticmethod
    def _compute_leftover(pool: List[int], subset: List[int]) -> List[int]:
        """Return the dice in `pool` not selected in `subset`.  Multiset subtraction."""
        # Work on multisets: for each die in subset, remove one occurrence from pool copy
        pool_copy = list(pool)
        for d in subset:
            try:
                pool_copy.remove(d)
            except ValueError:
                # This should not happen; ignore gracefully
                continue
        return pool_copy

    # -------------------------- Update Functions ---------------------------
    def update_get(self, dice_a: List[int], dice_b: List[int], my_bid: Bid, opp_bid: Bid, my_group: str) -> None:
        """
        Apply the result of a bidding phase.
        This function updates both our state and the opponent's state with
        the dice they receive and adjusts bidding scores accordingly.
        """
        # Determine which dice each player receives based on the resolved group
        if my_group == "A":
            self.my_state.add_dice(dice_a)
            self.opp_state.add_dice(dice_b)
        else:
            self.my_state.add_dice(dice_b)
            self.opp_state.add_dice(dice_a)
        # Apply bid results
        my_bid_ok = (my_bid.group == my_group)
        self.my_state.bid(my_bid_ok, my_bid.amount)
        opp_group_resolved = "A" if my_group == "B" else "B"
        opp_bid_ok = (opp_bid.group == opp_group_resolved)
        self.opp_state.bid(opp_bid_ok, opp_bid.amount)

    def update_put(self, put: DicePut) -> None:
        """Record our scoring decision and consume the selected dice."""
        self.my_state.use_dice(put)
        # Advance round number when scoring phase is completed
        self.round += 1

    def update_set(self, put: DicePut) -> None:
        """Record the opponent's scoring decision."""
        self.opp_state.use_dice(put)


def main() -> None:
    """
    Main loop implementing the I/O protocol.

    Reads commands from stdin and responds with appropriate bidding and
    scoring actions.  Maintains internal game state in the `Game` instance.
    """
    game = Game()
    # Variables to hold dice rolls and our last bid
    dice_a: List[int] = [0] * 5
    dice_b: List[int] = [0] * 5
    my_bid: Bid = Bid("", 0)
    while True:
        try:
            line = input().strip()
            if not line:
                continue
            parts = line.split()
            command = parts[0]
            if command == "READY":
                # Acknowledge readiness
                print("OK")
                sys.stdout.flush()
                continue
            if command == "ROLL":
                # Two bundles of 5 dice are presented
                str_a, str_b = parts[1], parts[2]
                dice_a = [int(c) for c in str_a]
                dice_b = [int(c) for c in str_b]
                my_bid = game.calculate_bid(dice_a, dice_b)
                print(f"BID {my_bid.group} {my_bid.amount}")
                sys.stdout.flush()
                continue
            if command == "GET":
                # Inform us of the outcome of the bid
                # Format: GET myGroup oppGroup oppAmount
                my_group, opp_group, opp_amount_str = parts[1], parts[2], parts[3]
                opp_amount = int(opp_amount_str)
                game.update_get(dice_a, dice_b, my_bid, Bid(opp_group, opp_amount), my_group)
                continue
            if command == "SCORE":
                # Our turn to select dice and category
                put = game.calculate_put()
                game.update_put(put)
                print(f"PUT {put.rule.name} {''.join(map(str, put.dice))}")
                sys.stdout.flush()
                continue
            if command == "SET":
                # Opponent's scoring decision; update opponent state
                rule_name, str_dice = parts[1], parts[2]
                opp_put = DicePut(DiceRule[rule_name], [int(c) for c in str_dice])
                game.update_set(opp_put)
                continue
            if command == "FINISH":
                # Game over
                break
            # Unexpected commands are ignored
        except EOFError:
            break


if __name__ == "__main__":
    main()