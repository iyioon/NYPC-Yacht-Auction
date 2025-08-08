"""
Production agent using Monte Carlo Tree Search (MCTS) with a trained neural
network to play the yacht dice bidding game.  This script follows the I/O
protocol described in `Instruction_eng.md` and relies only on the
`data.bin` file for network weights and the allowed libraries.  It
integrates the `Game` and `NNet` implementations from the `yacht`
package and uses an AlphaZero‑style MCTS to select actions.

Usage:
    python3 agent_mcts.py

You should ensure that `data.bin` is present in the working directory
containing the trained weights (produced by training with the
alpha‑zero‑general framework).  The script communicates via stdin/stdout
with the referee.  Time limits per move are enforced externally; this
implementation uses a fixed number of MCTS simulations for each
decision.
"""

from __future__ import annotations

import sys
import math
import numpy as np
import random
from typing import Dict, Tuple, List

# Import game and neural network from yacht package
from yacht.Game import Game
from yacht.NNet import NNetWrapper


class MCTS:
    """
    Monte Carlo Tree Search implementation adapted from alpha-zero-general.

    The tree is stored in dictionaries mapping state/action pairs to values.
    For each new state encountered, the neural network is used to obtain
    initial policy and value estimates.  Valid moves are masked using the
    game's `getValidMoves` method.  The search uses the UCB formula to
    balance exploration and exploitation.  After a predetermined number of
    simulations, the visit counts form a policy that guides the agent's
    move selection.
    """

    def __init__(self, game: Game, nnet: NNetWrapper, cpuct: float = 1.0) -> None:
        self.game = game
        self.nnet = nnet
        self.cpuct = cpuct
        # Qsa[(s,a)] = Q value for state s and action a
        self.Qsa: Dict[Tuple[str, int], float] = {}
        # Nsa[(s,a)] = number of times edge (s,a) was visited
        self.Nsa: Dict[Tuple[str, int], int] = {}
        # Ns[s] = number of times state s was visited
        self.Ns: Dict[str, int] = {}
        # Ps[s] = initial policy returned by neural network for state s
        self.Ps: Dict[str, np.ndarray] = {}
        # Es[s] = game end status for state s (0, 1, -1, small draw value)
        self.Es: Dict[str, float] = {}
        # Vs[s] = valid moves for state s as binary mask
        self.Vs: Dict[str, np.ndarray] = {}

    def get_action_prob(self, board: Dict, player: int, temp: float = 1.0, num_sims: int = 50) -> np.ndarray:
        """
        Run MCTS simulations starting from the given canonical board state
        and return a probability distribution over actions.  If `temp` is
        zero, the distribution will be a one‑hot at the most visited action.
        The number of simulations controls the computational budget.
        """
        for _ in range(num_sims):
            self.search(board, player)
        s = self.game.stringRepresentation(board)
        counts = np.array([self.Nsa.get((s, a), 0) for a in range(
            self.game.getActionSize())], dtype=np.float32)
        if temp == 0:
            # Deterministic: choose the move with the highest visit count
            best_actions = np.argwhere(counts == np.max(counts)).flatten()
            probs = np.zeros_like(counts)
            probs[random.choice(best_actions)] = 1.0
            return probs
        # Temperature > 0: apply softmax over log counts
        counts = counts ** (1.0 / temp)
        if counts.sum() == 0:
            counts = np.ones_like(counts)
        probs = counts / counts.sum()
        return probs

    def search(self, board: Dict, player: int) -> float:
        """
        Perform one MCTS simulation.  Returns the value of the current
        position from the perspective of the current player.  The board
        argument must be in canonical form (player 1's perspective).
        """
        s = self.game.stringRepresentation(board)
        # Check if game has ended
        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(board, player)
        if self.Es[s] != 0:
            # Terminal node: return negative value because we store values
            # from the perspective of the previous player
            return -self.Es[s]
        # If state not visited yet, expand it using the network
        if s not in self.Ps:
            # Use neural network to get initial policy and value
            policy, v = self.nnet.predict(board)
            valid_moves = self.game.getValidMoves(board, player)
            policy = policy * valid_moves  # mask invalid moves
            sum_p = policy.sum()
            if sum_p > 0:
                policy /= sum_p
            else:
                # If all valid moves were masked (rare), assign uniform prob
                policy = valid_moves / np.maximum(valid_moves.sum(), 1)
            self.Ps[s] = policy
            self.Vs[s] = valid_moves
            self.Ns[s] = 0
            # Value from the current player's perspective
            return -v
        # Select the move with maximum UCT value
        best_ucb = -float('inf')
        best_action = -1
        for a in range(self.game.getActionSize()):
            if self.Vs[s][a] == 0:
                continue
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * \
                    math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + 1e-8)
            if u > best_ucb:
                best_ucb = u
                best_action = a
        a = best_action
        # Recurse on the selected action
        next_board, next_player = self.game.getNextState(board, player, a)
        next_board = self.game.getCanonicalForm(next_board, next_player)
        v = self.search(next_board, next_player)
        # Update Qsa and Nsa
        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] *
                                self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        self.Ns[s] += 1
        return -v


def nearest_bid_index(bid_levels: List[int], amount: int) -> int:
    """
    Find the index in `bid_levels` (in thousands) that is closest to
    `amount // 1000`.  This is used to discretise the opponent's bid
    amount to our predefined levels.
    """
    target = amount // 1000
    best_idx = 0
    best_diff = abs(bid_levels[0] - target)
    for i, level in enumerate(bid_levels):
        diff = abs(level - target)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    return best_idx


# Mapping from category names in SET commands to category indices
_CATEGORY_MAP = {
    'ONE': 0,
    'TWO': 1,
    'THREE': 2,
    'FOUR': 3,
    'FIVE': 4,
    'SIX': 5,
    'CHOICE': 6,
    'FOUR_OF_A_KIND': 7,
    'FULL_HOUSE': 8,
    'SMALL_STRAIGHT': 9,
    'LARGE_STRAIGHT': 10,
    'YACHT': 11,
}


def find_subset_index(dice_pool: List[int], selected: List[int], subsets: List[Tuple[int, ...]]) -> int:
    """
    Given a player's current dice pool and a list of selected dice values,
    find the subset index in `subsets` that corresponds to selecting
    positions in the pool that match the multiset of selected values.
    Returns -1 if no matching subset is found.
    """
    n = len(dice_pool)
    # Generate all combinations of positions of size len(selected)
    # We want exactly 5 dice selected
    required = len(selected)
    # Sort the selected values for multiset comparison
    selected_sorted = sorted(selected)
    for idx, subset in enumerate(subsets):
        if max(subset) >= n:
            continue
        if len(subset) != required:
            continue
        # Extract dice values at these positions
        vals = [dice_pool[i] for i in subset]
        if sorted(vals) == selected_sorted:
            return idx
    return -1


def main() -> None:
    game = Game()
    nnet = NNetWrapper(game)
    # Load trained model from data.bin
    try:
        nnet.load_checkpoint('.', 'data.bin')
    except Exception:
        # If loading fails, we proceed with an uninitialised network
        pass
    # Instantiate MCTS with reasonable cpuct
    mcts = MCTS(game, nnet, cpuct=1.0)
    # Initialise game state
    board = game.getInitBoard()
    player = 1  # from our perspective
    # Convert to canonical form before MCTS
    board = game.getCanonicalForm(board, player)
    # Track last actions if needed
    while True:
        try:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            command = parts[0]
            if command == 'READY':
                print('OK')
                sys.stdout.flush()
                continue
            if command == 'ROLL':
                # Update bundles with new dice
                str_a, str_b = parts[1], parts[2]
                board['bundle_a'] = [int(c) for c in str_a]
                board['bundle_b'] = [int(c) for c in str_b]
                # MCTS decision on bidding action
                # Use temperature 0 to choose deterministically
                probs = mcts.get_action_prob(
                    board, player, temp=0, num_sims=50)
                # Mask to bidding actions
                valid = game.getValidMoves(board, player)
                probs = probs * valid
                if probs.sum() == 0:
                    # fallback to random valid bidding action
                    actions = np.where(valid[: game.num_bid_actions] == 1)[0]
                    action = int(random.choice(actions))
                else:
                    action = int(np.argmax(probs))
                # Decode selected action to group and bid amount
                bundle_choice = action // len(game.bid_levels)
                bid_idx = action % len(game.bid_levels)
                bid_amount = game.bid_levels[bid_idx] * 1000
                # Update internal board using getNextState
                board, player = game.getNextState(board, player, action)
                board = game.getCanonicalForm(board, player)
                # Output bid
                group_char = 'A' if bundle_choice == 0 else 'B'
                print(f'BID {group_char} {bid_amount}')
                sys.stdout.flush()
                continue
            if command == 'GET':
                # Notification of bidding outcome
                # Format: GET myGroup oppGroup oppAmount
                my_group_char, opp_group_char, opp_amount_str = parts[1], parts[2], parts[3]
                opp_amount = int(opp_amount_str)
                # Determine opponent's action
                # Map group char to choice 0(A) or 1(B)
                opp_choice = 0 if opp_group_char == 'A' else 1
                # Determine closest bid level index
                opp_bid_idx = nearest_bid_index(game.bid_levels, opp_amount)
                opponent_action = opp_choice * \
                    len(game.bid_levels) + opp_bid_idx
                # Update board/state with opponent's bidding action
                board, player = game.getNextState(
                    board, player, opponent_action)
                board = game.getCanonicalForm(board, player)
                continue
            if command == 'SCORE':
                # Our turn to select scoring action
                probs = mcts.get_action_prob(
                    board, player, temp=0, num_sims=50)
                # Mask to scoring actions
                valid = game.getValidMoves(board, player)
                probs = probs * valid
                # Choose the highest probability scoring action
                if probs.sum() == 0:
                    # fallback: choose first valid scoring action
                    scoring_actions = np.where(valid == 1)[0]
                    # Exclude bidding actions
                    scoring_actions = [
                        a for a in scoring_actions if a >= game.num_bid_actions]
                    action = int(random.choice(scoring_actions))
                else:
                    action = int(np.argmax(probs))
                # Decode scoring action to category and dice values for output
                scoring_action = action - game.num_bid_actions
                cat = scoring_action // game.num_dice_subsets
                subset_idx = scoring_action % game.num_dice_subsets
                subset = game.subsets[subset_idx]
                # Get current player's dice (always dice0 in canonical form)
                dice_pool = board['dice0']

                # Validate and adjust subset to ensure it's compatible with dice pool
                if max(subset) >= len(dice_pool):
                    # Find a valid subset for current pool size
                    valid_subsets = [
                        s for s in game.subsets if max(s) < len(dice_pool)]
                    if valid_subsets:
                        subset = valid_subsets[0]
                        # Recalculate subset_idx to match the corrected subset
                        subset_idx = game.subset_to_index[subset]
                    else:
                        # Pool has fewer than 5 dice, use all available
                        subset = tuple(range(len(dice_pool)))
                        if subset in game.subset_to_index:
                            subset_idx = game.subset_to_index[subset]

                chosen_dice_vals = [dice_pool[i] for i in subset]
                # Update board via getNextState
                board, player = game.getNextState(board, player, action)
                board = game.getCanonicalForm(board, player)
                # Output scoring decision
                # Map category index back to name
                category_name = [
                    k for k, v in _CATEGORY_MAP.items() if v == cat][0]
                print(
                    f'PUT {category_name} {"".join(map(str, chosen_dice_vals))}')
                sys.stdout.flush()
                continue
            if command == 'SET':
                # Opponent's scoring decision
                rule_name, dice_str = parts[1], parts[2]
                opp_cat = _CATEGORY_MAP[rule_name]
                selected_dice = [int(c) for c in dice_str]
                # Determine subset index for opponent's dice selection
                # In canonical board, opponent's dice are stored in dice1
                opp_dice_pool = board['dice1']
                subset_idx = find_subset_index(
                    opp_dice_pool, selected_dice, game.subsets)
                if subset_idx < 0:
                    # If no matching subset found, choose first possible subset (fallback)
                    # Find any subset whose values match selected dice length
                    subset_idx = 0
                opponent_action = game.num_bid_actions + \
                    opp_cat * game.num_dice_subsets + subset_idx
                board, player = game.getNextState(
                    board, player, opponent_action)
                board = game.getCanonicalForm(board, player)
                continue
            if command == 'FINISH':
                break
        except EOFError:
            break


if __name__ == '__main__':
    main()
