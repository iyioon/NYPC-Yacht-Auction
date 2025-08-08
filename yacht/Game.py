"""
Game wrapper for the yacht dice bidding game suitable for the alpha-zero-general
framework.

The alpha-zero-general framework expects games to implement a standard set
of methods: getInitBoard(), getBoardSize(), getActionSize(), getNextState(),
getValidMoves(), getGameEnded(), getCanonicalForm(), getSymmetries() and
stringRepresentation().  This file provides a skeleton for those methods
tailored to the 13-round yacht dice game.  Many details must be completed
before training (see TODOs).  See the accompanying report for guidance on
encoding the state and action spaces.
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np


import itertools
import random
import copy


class Game:
    def __init__(self) -> None:
        """
        Initialise parameters for the yacht game.  You can define constants
        such as the number of rounds, number of categories and discretisation
        levels for bids here.
        """
        # Total number of rounds (13 in the standard game)
        self.num_rounds = 13
        # Number of categories (12)
        self.num_categories = 12
        # Discretised bid levels.  For example, allow bids of 0, 10, 20, …, 100.
        # During integration you should multiply these values by 1000 to map to
        # point deductions/additions.
        self.bid_levels = [0, 10, 20, 30, 40, 50, 60, 80, 100]
        # Compute the total number of bidding actions (2 bundles × bid levels)
        self.num_bid_actions = 2 * len(self.bid_levels)
        # Number of possible 5‑dice subsets of a 10‑dice hand
        self.num_dice_subsets = 252  # C(10,5)
        # Total number of scoring actions = categories × subsets
        self.num_score_actions = self.num_categories * self.num_dice_subsets
        # Total action size = bidding actions + scoring actions
        self.n_actions = self.num_bid_actions + self.num_score_actions

        # Precompute all 5‑dice subsets of 10 positions and mapping to indices
        self.subsets: List[Tuple[int, ...]] = list(itertools.combinations(range(10), 5))
        self.subset_to_index = {subset: i for i, subset in enumerate(self.subsets)}

        # Vector length for board representation
        # Breakdown: round (13), phase (2), dice0 (70), dice1 (70), bundles (70),
        # used0 (12), used1 (12), scores0 (12), scores1 (12), bid_score0 (1),
        # bid_score1 (1), pending (13), scoring_step (1) => 289
        self.vector_length = 289

    def getInitBoard(self) -> np.ndarray:
        """
        Return an initial state representation.  The state includes the
        round number, dice currently held by both players (none at the start),
        scores recorded so far, whether categories have been used and
        placeholders for bundles shown in the bidding phase.

        The exact shape and content of the returned array depend on your
        encoding choice.  A simple approach is to return a 1D numpy array of
        zeros whose length matches the flattened state tensor described in
        the report.  You must ensure consistency between this representation
        and the neural network input.
        """
        """
        Create the initial board state.  The board is represented as a
        dictionary with explicit fields rather than a numpy array.  The
        canonical and vector forms are derived from this dictionary.
        """
        # Generate two random bundles of five dice each for the first bidding
        bundle_a = [random.randint(1, 6) for _ in range(5)]
        bundle_b = [random.randint(1, 6) for _ in range(5)]
        board = {
            'round': 1,
            'phase': 'bid',         # 'bid' or 'score'
            'player': 0,            # 0 or 1 (0 starts bidding)
            'dice0': [],            # dice pool for player 0
            'dice1': [],            # dice pool for player 1
            'scores0': [0] * self.num_categories,  # scores for player 0 (0 if unused)
            'scores1': [0] * self.num_categories,  # scores for player 1
            'used0': [False] * self.num_categories,  # used flags for player 0
            'used1': [False] * self.num_categories,  # used flags for player 1
            'bid_score0': 0,       # bidding score for player 0
            'bid_score1': 0,       # bidding score for player 1
            'bundle_a': bundle_a,   # current bundle A
            'bundle_b': bundle_b,   # current bundle B
            'pending': None,       # pending bid action (dict) or None
            'pending_scoring': None  # pending scoring player index (0 or 1) or None
        }
        return board

    def getBoardSize(self) -> Tuple[int, int, int]:
        """
        Return a tuple (channels, height, width) describing the dimensions
        of the state tensor expected by the neural network.  If you opt for
        a 1D flat vector representation, you can return (1, 1, length).
        """
        # TODO: adjust these values to match your state encoding.
        return (1, 1, self.vector_length)

    def getActionSize(self) -> int:
        """Return the total number of discrete actions available in the game."""
        return self.n_actions

    def getNextState(self, board: np.ndarray, player: int, action: int) -> Tuple[np.ndarray, int]:
        """
        Given a current state `board`, the current player (1 for the agent,
        -1 for the opponent) and an action index, return the next state and
        the next player.  This method must handle both bidding and scoring
        phases, update dice pools, scores and round number, and incorporate
        chance events (dice rolls and tie breaks).  For self‑play you can
        sample chance outcomes.

        Parameters:
            board (np.ndarray): current state representation.
            player (int): 1 or -1 indicating which perspective the board is
                presented from.  The canonical form always presents the board
                from the perspective of the player to move.
            action (int): integer in [0, n_actions) representing a discrete
                bidding or scoring action.

        Returns:
            (next_board, next_player): the new board representation and the
                player to move next.  If the round ended and roles swap,
                `next_player` will be -player.
        """
        # Convert board to our internal dict representation if necessary
        # In this implementation, board is always a dict created by getInitBoard
        state = copy.deepcopy(board)
        # Validate player index (0 or 1).  The canonical form will always call
        # getNextState with player=1.  We ignore `player` here because the
        # state carries the current player.
        current_player = state['player']

        # Bidding phase
        if state['phase'] == 'bid':
            # Decode bidding action: 0..17 => (bundle_choice, bid_level_index)
            # bundle_choice 0 -> A, 1 -> B
            # bid_level_index 0..len(self.bid_levels)-1
            bundle_choice = action // len(self.bid_levels)
            bid_index = action % len(self.bid_levels)
            bid_amount = self.bid_levels[bid_index] * 1000
            # If there is no pending bid, record this bid and pass turn
            if state['pending'] is None:
                state['pending'] = {
                    'player': current_player,
                    'choice': bundle_choice,
                    'bid_amount': bid_amount
                }
                # Switch turn to other player
                state['player'] = 1 - current_player
                return state, -player  # Next player indicator flips sign
            # There is a pending bid: resolve both bids
            first_bid = state['pending']
            second_bid = {
                'player': current_player,
                'choice': bundle_choice,
                'bid_amount': bid_amount
            }
            # Determine targeted bundles for each player (0/1 -> A/B)
            targeted = [None, None]  # 0 for player0, 1 for player1
            targeted[first_bid['player']] = first_bid['choice']
            targeted[second_bid['player']] = second_bid['choice']
            # Determine winner of contested bundle if both target same
            assignment = [None, None]  # assigned bundle for each player (0 for A, 1 for B)
            if targeted[0] != targeted[1]:
                # Different targets: each gets what they chose
                assignment[0] = targeted[0]
                assignment[1] = targeted[1]
            else:
                # Same target: compare bid amounts
                amt0 = first_bid['bid_amount'] if first_bid['player'] == 0 else second_bid['bid_amount']
                amt1 = first_bid['bid_amount'] if first_bid['player'] == 1 else second_bid['bid_amount']
                # Determine winner: higher amount gets targeted; lower gets opposite
                if amt0 > amt1:
                    winner = 0
                elif amt1 > amt0:
                    winner = 1
                else:
                    # Tie: pick random winner
                    winner = random.randint(0, 1)
                assignment[winner] = targeted[0]
                assignment[1 - winner] = 1 - targeted[0]
            # Distribute bundles and update bid scores
            for i in (0, 1):
                chosen_bundle = assignment[i]
                # Add dice to player's pool
                if chosen_bundle == 0:
                    state[f'dice{i}'].extend(state['bundle_a'])
                else:
                    state[f'dice{i}'].extend(state['bundle_b'])
            # Adjust bid scores: subtract amount if got targeted, add otherwise
            for i in (0, 1):
                targeted_bundle = targeted[i]
                assigned_bundle = assignment[i]
                # Retrieve bid amount for this player
                amt = first_bid['bid_amount'] if first_bid['player'] == i else second_bid['bid_amount']
                if assigned_bundle == targeted_bundle:
                    state[f'bid_score{i}'] -= amt
                else:
                    state[f'bid_score{i}'] += amt
            # Clear pending bid
            state['pending'] = None
            # After bidding resolution, determine next phase
            if state['round'] == 1:
                # Round 1 has no scoring; move directly to next bidding round
                state['round'] = 2
                state['phase'] = 'bid'
                state['player'] = 0  # Start next round with player 0 (arbitrary)
                # Generate new bundles
                state['bundle_a'] = [random.randint(1, 6) for _ in range(5)]
                state['bundle_b'] = [random.randint(1, 6) for _ in range(5)]
            elif 2 <= state['round'] <= 12:
                # Prepare for scoring phase
                state['phase'] = 'score'
                # Determine who scores first: alternate by round to keep fairness
                state['player'] = state['round'] % 2  # player 0 scores first on even rounds
                state['pending_scoring'] = None
            # Return next state and next player indicator (-player flips sign)
            return state, -player

        # Scoring phase
        if state['phase'] == 'score':
            # Determine offset for scoring actions
            scoring_action = action - self.num_bid_actions
            # Determine current player index
            p = current_player
            # Ensure current player has not yet scored in this round
            if state['pending_scoring'] is not None and state['pending_scoring'] == p:
                # This should not happen; current player already scored
                return state, -player
            # Get player's dice and used categories
            dice = state[f'dice{p}']
            used = state[f'used{p}']
            # Decode category and subset
            cat = scoring_action // self.num_dice_subsets
            subset_index = scoring_action % self.num_dice_subsets
            # Validate category is unused
            if used[cat]:
                # Invalid move; return same state (will be masked out in getValidMoves)
                return state, -player
            # Validate subset fits within current dice length
            # Determine dice positions available (0..len(dice)-1)
            n = len(dice)
            if n < 5:
                # Should not happen until final round; treat subset as first 5 dice
                chosen_positions = tuple(range(n))
            else:
                # Check if subset_index corresponds to positions < n
                chosen_positions = self.subsets[subset_index]
                if max(chosen_positions) >= n:
                    # Invalid subset for current hand size
                    return state, -player
            # Compute chosen dice values
            chosen_dice = [dice[i] for i in chosen_positions]
            # Remove chosen dice from player's hand
            for i in sorted(chosen_positions, reverse=True):
                # Remove by index; use reverse order to avoid shifting indices
                del dice[i]
            # Compute score for this category
            # Use sample-code's scoring logic
            score = self._calculate_category_score(cat, chosen_dice)
            # Record score and mark category used
            state[f'scores{p}'][cat] = score
            state[f'used{p}'][cat] = True
            # Mark that this player has scored
            if state['pending_scoring'] is None:
                state['pending_scoring'] = p
                # Switch to other player's scoring turn
                state['player'] = 1 - p
                return state, -player
            else:
                # Both players have scored; finish round
                state['pending_scoring'] = None
                # Increment round
                state['round'] += 1
                if state['round'] <= 12:
                    # Start bidding next round
                    state['phase'] = 'bid'
                    state['player'] = 0  # Start bidding with player 0
                    # Generate new bundles
                    state['bundle_a'] = [random.randint(1, 6) for _ in range(5)]
                    state['bundle_b'] = [random.randint(1, 6) for _ in range(5)]
                elif state['round'] == 13:
                    # Final round: no more bidding; just scoring
                    state['phase'] = 'score'
                    state['player'] = state['round'] % 2  # alternate starting scorer
                else:
                    # Game ends after final scoring
                    state['phase'] = 'end'
                return state, -player

        # If phase is 'end', return same state
        return state, 0

    def getValidMoves(self, board: np.ndarray, player: int) -> np.ndarray:
        """
        Return a binary vector of length `n_actions` indicating which actions
        are legal in the given state.  During the bidding phase only the
        bidding actions corresponding to both bundles and available bid
        levels are legal.  During scoring only the scoring actions for
        unused categories and available dice subsets are legal.
        """
        # Prepare valid moves array
        valid = np.zeros(self.n_actions, dtype=np.int8)
        state = board
        # If game ended, no valid moves
        if state['phase'] == 'end':
            return valid
        if state['phase'] == 'bid':
            # All bidding actions are valid on a bidding turn
            # There are no restrictions on repeating bids or bundles
            valid[: self.num_bid_actions] = 1
            return valid
        if state['phase'] == 'score':
            # Determine current player index
            p = state['player']
            # Determine dice length
            dice = state[f'dice{p}']
            n = len(dice)
            # Determine which categories are unused
            used = state[f'used{p}']
            for cat in range(self.num_categories):
                if used[cat]:
                    continue
                # Determine subsets of positions
                if n < 5:
                    # Use a single subset of all positions
                    subsets_indices = [self.subset_to_index[tuple(range(n))]]
                else:
                    # Use precomputed subsets whose max index < n
                    subsets_indices = [i for i, s in enumerate(self.subsets) if max(s) < n]
                # Mark corresponding actions as valid
                base = self.num_bid_actions + cat * self.num_dice_subsets
                for sub_i in subsets_indices:
                    valid[base + sub_i] = 1
            return valid
        # If unknown phase
        return valid

    def getGameEnded(self, board: np.ndarray, player: int) -> float:
        """
        Return the game result from the perspective of `player`.

        Returns:
            0 if the game is not yet finished;
            1 if the current player wins;
            -1 if the current player loses;
            a small value (e.g. 1e-4) for a draw.
        """
        state = board
        # Game ends if phase is 'end' or round > 13
        if state['phase'] != 'end' and state['round'] <= self.num_rounds:
            return 0.0
        # Compute total scores for both players
        total = []
        for i in (0, 1):
            basic = sum(state[f'scores{i}'][0:6])
            bonus = 35000 if basic >= 63000 else 0
            combo = sum(state[f'scores{i}'][6:12])
            total_score = basic + bonus + combo + state[f'bid_score{i}']
            total.append(total_score)
        # Determine result from perspective of `player` (1 or -1).  In this game
        # we consider player index 0 as 1 and index 1 as -1.  We map
        # `player` to this indexing.
        # canonical form uses current player=1, so we always compare total[0] vs total[1]
        if total[0] > total[1]:
            return 1.0
        elif total[0] < total[1]:
            return -1.0
        else:
            return 1e-4

    def getCanonicalForm(self, board: np.ndarray, player: int) -> np.ndarray:
        """
        Return the canonical form of the board.  In the canonical form the
        state is always presented from the perspective of the player to move
        (player = 1).  If `player` is -1, you must swap the roles of the
        two players in the state encoding so that neural network training
        remains consistent.
        """
        # The canonical form always places the player to move as player 0.
        # If board['player'] is 0, return board unchanged.  If 1, swap
        # players' information.
        # Deep copy to avoid modifying original board
        state = copy.deepcopy(board)
        if state['player'] == 0:
            # Already canonical
            return state
        # Swap players 0 and 1 fields
        for key in ['dice', 'scores', 'used', 'bid_score']:
            state[f'{key}0'], state[f'{key}1'] = state[f'{key}1'], state[f'{key}0']
        # Adjust pending bid: swap player index
        if state['pending'] is not None:
            state['pending'] = {
                'player': 1 - state['pending']['player'],
                'choice': state['pending']['choice'],
                'bid_amount': state['pending']['bid_amount']
            }
        # Adjust pending scoring
        if state['pending_scoring'] is not None:
            state['pending_scoring'] = 1 - state['pending_scoring']
        # Canonical player always 0
        state['player'] = 0
        return state

    def getSymmetries(self, board: np.ndarray, pi: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Return all symmetrical equivalents of the board and action
        probability vector.  The yacht game has no geometric symmetries,
        but if you find meaningful transformations (e.g. swapping the two
        bundles before bidding), you can implement them here.
        """
        # No symmetries by default
        return [(board, pi)]

    def stringRepresentation(self, board: np.ndarray) -> str:
        """
        Return a unique string representation of the board for hashing in the
        MCTS.  Ensure that different game states produce different strings.
        """
        # Convert the board to a unique string via its vector representation
        vec = self._board_to_vector(board)
        return ''.join(map(lambda x: format(int(x*1000), '03d'), vec))

    # --------------------------- Helper Functions ---------------------------
    def _calculate_category_score(self, cat: int, dice: List[int]) -> int:
        """Compute the score for a given category index and chosen dice."""
        # Basic categories
        if cat == 0:
            return sum(d for d in dice if d == 1) * 1000
        if cat == 1:
            return sum(d for d in dice if d == 2) * 1000
        if cat == 2:
            return sum(d for d in dice if d == 3) * 1000
        if cat == 3:
            return sum(d for d in dice if d == 4) * 1000
        if cat == 4:
            return sum(d for d in dice if d == 5) * 1000
        if cat == 5:
            return sum(d for d in dice if d == 6) * 1000
        # Choice
        if cat == 6:
            return sum(dice) * 1000
        # Four of a kind
        if cat == 7:
            ok = any(dice.count(i) >= 4 for i in range(1, 7))
            return sum(dice) * 1000 if ok else 0
        # Full house
        if cat == 8:
            pair = triple = False
            for i in range(1, 7):
                cnt = dice.count(i)
                if cnt == 2 or cnt == 5:
                    pair = True
                if cnt == 3 or cnt == 5:
                    triple = True
            return sum(dice) * 1000 if pair and triple else 0
        # Small straight
        if cat == 9:
            e = [dice.count(i) > 0 for i in range(1, 7)]
            ok = (e[0] and e[1] and e[2] and e[3]) or (e[1] and e[2] and e[3] and e[4]) or (e[2] and e[3] and e[4] and e[5])
            return 15000 if ok else 0
        # Large straight
        if cat == 10:
            e = [dice.count(i) > 0 for i in range(1, 7)]
            ok = (e[0] and e[1] and e[2] and e[3] and e[4]) or (e[1] and e[2] and e[3] and e[4] and e[5])
            return 30000 if ok else 0
        # Yacht
        if cat == 11:
            ok = any(dice.count(i) == 5 for i in range(1, 7))
            return 50000 if ok else 0
        return 0

    def _board_to_vector(self, board: dict) -> np.ndarray:
        """
        Flatten the board dictionary into a 1D numpy array of fixed length
        suitable for neural network input.  The canonical form should
        already place the player to move as player 0.  Values are normalised
        to roughly [0,1] where appropriate.
        """
        vec = np.zeros(self.vector_length, dtype=np.float32)
        idx = 0
        # Round one-hot (13 bits)
        for r in range(1, self.num_rounds + 1):
            vec[idx] = 1.0 if board['round'] == r else 0.0
            idx += 1
        # Phase one-hot (2 bits)
        if board['phase'] == 'bid':
            vec[idx] = 1.0
        idx += 1
        if board['phase'] == 'score':
            vec[idx] = 1.0
        idx += 1
        # Player 0 dice (10 positions × 7 flags)
        for pos in range(10):
            if pos < len(board['dice0']):
                val = board['dice0'][pos]
                # one-hot for values 1-6
                vec[idx + val - 1] = 1.0
            else:
                # empty flag at last position
                vec[idx + 6] = 1.0
            idx += 7
        # Player 1 dice
        for pos in range(10):
            if pos < len(board['dice1']):
                val = board['dice1'][pos]
                vec[idx + val - 1] = 1.0
            else:
                vec[idx + 6] = 1.0
            idx += 7
        # Bundles: two bundles of 5 dice
        for bundle in ['bundle_a', 'bundle_b']:
            dice_list = board[bundle]
            for pos in range(5):
                if pos < len(dice_list):
                    val = dice_list[pos]
                    vec[idx + val - 1] = 1.0
                else:
                    vec[idx + 6] = 1.0
                idx += 7
        # Player 0 used categories (12 bits)
        for used in board['used0']:
            vec[idx] = 1.0 if used else 0.0
            idx += 1
        # Player 1 used categories
        for used in board['used1']:
            vec[idx] = 1.0 if used else 0.0
            idx += 1
        # Player 0 scores (12 values, scaled)
        for s in board['scores0']:
            vec[idx] = s / 100000.0  # scale scores to roughly [0,1]
            idx += 1
        # Player 1 scores
        for s in board['scores1']:
            vec[idx] = s / 100000.0
            idx += 1
        # Player 0 bid score (scaled)
        vec[idx] = board['bid_score0'] / 100000.0
        idx += 1
        # Player 1 bid score (scaled)
        vec[idx] = board['bid_score1'] / 100000.0
        idx += 1
        # Pending information (13 bits)
        if board['pending'] is not None:
            vec[idx] = 1.0  # pending flag
            idx += 1
            # Group one-hot (2 bits)
            grp = board['pending']['choice']
            vec[idx + grp] = 1.0
            idx += 2
            # Bid index one-hot (9 bits)
            # Determine bid index corresponding to bid amount
            try:
                bidx = self.bid_levels.index(board['pending']['bid_amount'] // 1000)
            except ValueError:
                bidx = 0
            for i in range(len(self.bid_levels)):
                vec[idx + i] = 1.0 if i == bidx else 0.0
            idx += len(self.bid_levels)
            # Pending belongs to opponent? 1 if pending['player'] != current player 0
            vec[idx] = 1.0 if board['pending']['player'] != 0 else 0.0
            idx += 1
        else:
            # No pending: set flag and skip bits
            vec[idx] = 0.0
            idx += 1
            # group bits
            idx += 2
            # bid index bits
            idx += len(self.bid_levels)
            # pending_is_opponent bit
            vec[idx] = 0.0
            idx += 1
        # Scoring step flag: 1 if pending_scoring is not None
        vec[idx] = 1.0 if board['pending_scoring'] is not None else 0.0
        idx += 1
        # Sanity check
        assert idx == self.vector_length, f"Vector length mismatch: {idx} vs {self.vector_length}"
        return vec
