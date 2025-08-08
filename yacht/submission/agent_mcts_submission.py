"""
Production MCTS agent for yacht dice bidding game - Tournament Submission Version

This is a self-contained agent that uses Monte Carlo Tree Search (MCTS) with a 
trained neural network to play the yacht dice bidding game. It follows the I/O 
protocol described in the game instructions and loads the trained model from 
'data.bin' in the same directory as this script.

For tournament submission, place this file alongside 'data.bin' containing the 
trained neural network weights.

Usage:
    python3 agent_mcts_submission.py

The agent communicates via stdin/stdout with the referee and uses a fixed number 
of MCTS simulations for each decision to respect time limits.
"""

import sys
import os
import math
import random
import copy
import itertools
from typing import Dict, Tuple, List, Optional
import numpy as np

# Neural network and PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim


# ========================== GAME IMPLEMENTATION ==========================

class Game:
    """Yacht dice game implementation for AlphaZero framework."""

    def __init__(self) -> None:
        self.num_rounds = 13
        self.num_categories = 12
        self.bid_levels = [0, 10, 20, 30, 40, 50, 60, 80, 100]
        self.num_bid_actions = 2 * len(self.bid_levels)
        self.num_dice_subsets = 252  # C(10,5)
        self.num_score_actions = self.num_categories * self.num_dice_subsets
        self.n_actions = self.num_bid_actions + self.num_score_actions

        # Precompute all 5-dice subsets
        self.subsets: List[Tuple[int, ...]] = list(
            itertools.combinations(range(10), 5))
        self.subset_to_index = {subset: i for i,
                                subset in enumerate(self.subsets)}
        self.vector_length = 289

    def getInitBoard(self) -> Dict:
        """Return initial game state."""
        bundle_a = [random.randint(1, 6) for _ in range(5)]
        bundle_b = [random.randint(1, 6) for _ in range(5)]
        return {
            'round': 1,
            'phase': 'bid',
            'player': 0,
            'dice0': [],
            'dice1': [],
            'scores0': [0] * self.num_categories,
            'scores1': [0] * self.num_categories,
            'used0': [False] * self.num_categories,
            'used1': [False] * self.num_categories,
            'bid_score0': 0,
            'bid_score1': 0,
            'bundle_a': bundle_a,
            'bundle_b': bundle_b,
            'pending': None,
            'pending_scoring': None
        }

    def getBoardSize(self) -> Tuple[int, int, int]:
        """Return board dimensions for neural network."""
        return (1, 1, self.vector_length)

    def getActionSize(self) -> int:
        """Return total number of possible actions."""
        return self.n_actions

    def getNextState(self, board: Dict, player: int, action: int) -> Tuple[Dict, int]:
        """Apply action and return next state."""
        state = copy.deepcopy(board)
        current_player = state['player']

        if state['phase'] == 'bid':
            # Decode bidding action
            bundle_choice = action // len(self.bid_levels)
            bid_index = action % len(self.bid_levels)
            bid_amount = self.bid_levels[bid_index] * 1000

            if state['pending'] is None:
                # First bid
                state['pending'] = {
                    'player': current_player,
                    'choice': bundle_choice,
                    'bid_amount': bid_amount
                }
                state['player'] = 1 - current_player
                return state, -player

            # Resolve both bids
            first_bid = state['pending']
            second_bid = {
                'player': current_player,
                'choice': bundle_choice,
                'bid_amount': bid_amount
            }

            # Determine bundle assignment
            targeted = [None, None]
            targeted[first_bid['player']] = first_bid['choice']
            targeted[second_bid['player']] = second_bid['choice']

            assignment = [None, None]
            if targeted[0] != targeted[1]:
                assignment[0] = targeted[0]
                assignment[1] = targeted[1]
            else:
                amt0 = first_bid['bid_amount'] if first_bid['player'] == 0 else second_bid['bid_amount']
                amt1 = first_bid['bid_amount'] if first_bid['player'] == 1 else second_bid['bid_amount']
                if amt0 > amt1:
                    winner = 0
                elif amt1 > amt0:
                    winner = 1
                else:
                    winner = random.randint(0, 1)
                assignment[winner] = targeted[0]
                assignment[1 - winner] = 1 - targeted[0]

            # Distribute dice and update scores
            for i in (0, 1):
                chosen_bundle = assignment[i]
                if chosen_bundle == 0:
                    state[f'dice{i}'].extend(state['bundle_a'])
                else:
                    state[f'dice{i}'].extend(state['bundle_b'])

                targeted_bundle = targeted[i]
                assigned_bundle = assignment[i]
                amt = first_bid['bid_amount'] if first_bid['player'] == i else second_bid['bid_amount']
                if assigned_bundle == targeted_bundle:
                    state[f'bid_score{i}'] -= amt
                else:
                    state[f'bid_score{i}'] += amt

            state['pending'] = None

            # Determine next phase
            if state['round'] == 1:
                state['round'] = 2
                state['phase'] = 'bid'
                state['player'] = 0
                state['bundle_a'] = [random.randint(1, 6) for _ in range(5)]
                state['bundle_b'] = [random.randint(1, 6) for _ in range(5)]
            elif 2 <= state['round'] <= 12:
                state['phase'] = 'score'
                state['player'] = state['round'] % 2
                state['pending_scoring'] = None

            return state, -player

        if state['phase'] == 'score':
            # Decode scoring action
            scoring_action = action - self.num_bid_actions
            p = current_player

            if state['pending_scoring'] is not None and state['pending_scoring'] == p:
                return state, -player

            dice = state[f'dice{p}']
            used = state[f'used{p}']

            cat = scoring_action // self.num_dice_subsets
            subset_index = scoring_action % self.num_dice_subsets

            if used[cat]:
                return state, -player

            n = len(dice)
            if n < 5:
                chosen_positions = tuple(range(n))
            else:
                chosen_positions = self.subsets[subset_index]
                if max(chosen_positions) >= n:
                    return state, -player

            chosen_dice = [dice[i] for i in chosen_positions]

            # Remove chosen dice
            for i in sorted(chosen_positions, reverse=True):
                del dice[i]

            # Calculate and record score
            score = self._calculate_category_score(cat, chosen_dice)
            state[f'scores{p}'][cat] = score
            state[f'used{p}'][cat] = True

            if state['pending_scoring'] is None:
                state['pending_scoring'] = p
                state['player'] = 1 - p
                return state, -player
            else:
                state['pending_scoring'] = None
                state['round'] += 1
                if state['round'] <= 12:
                    state['phase'] = 'bid'
                    state['player'] = 0
                    state['bundle_a'] = [
                        random.randint(1, 6) for _ in range(5)]
                    state['bundle_b'] = [
                        random.randint(1, 6) for _ in range(5)]
                elif state['round'] == 13:
                    state['phase'] = 'score'
                    state['player'] = state['round'] % 2
                else:
                    state['phase'] = 'end'
                return state, -player

        return state, 0

    def getValidMoves(self, board: Dict, player: int) -> np.ndarray:
        """Return binary vector of valid moves."""
        valid = np.zeros(self.n_actions, dtype=np.int8)
        state = board

        if state['phase'] == 'end':
            return valid

        if state['phase'] == 'bid':
            valid[:self.num_bid_actions] = 1
            return valid

        if state['phase'] == 'score':
            p = state['player']
            dice = state[f'dice{p}']
            n = len(dice)
            used = state[f'used{p}']

            for cat in range(self.num_categories):
                if used[cat]:
                    continue

                if n < 5:
                    # For fewer than 5 dice, we can only use subset (0,1,2,...,n-1)
                    # Find this subset in our precomputed list or use index 0 as fallback
                    target_subset = tuple(range(n))
                    if target_subset in self.subset_to_index:
                        subsets_indices = [self.subset_to_index[target_subset]]
                    else:
                        # Use first available subset as fallback
                        subsets_indices = [0]
                elif n == 5:
                    # Exactly 5 dice, use subset (0,1,2,3,4)
                    subsets_indices = [self.subset_to_index[(0, 1, 2, 3, 4)]]
                else:
                    # More than 5 dice, use all valid subsets
                    subsets_indices = [i for i, s in enumerate(
                        self.subsets) if max(s) < n]

                base = self.num_bid_actions + cat * self.num_dice_subsets
                for sub_i in subsets_indices:
                    valid[base + sub_i] = 1
            return valid

        return valid

    def getGameEnded(self, board: Dict, player: int) -> float:
        """Return game result from current player's perspective."""
        state = board
        if state['phase'] != 'end' and state['round'] <= self.num_rounds:
            return 0.0

        total = []
        for i in (0, 1):
            basic = sum(state[f'scores{i}'][0:6])
            bonus = 35000 if basic >= 63000 else 0
            combo = sum(state[f'scores{i}'][6:12])
            total_score = basic + bonus + combo + state[f'bid_score{i}']
            total.append(total_score)

        if total[0] > total[1]:
            return 1.0
        elif total[0] < total[1]:
            return -1.0
        else:
            return 1e-4

    def getCanonicalForm(self, board: Dict, player: int) -> Dict:
        """Return canonical form with current player as player 0."""
        state = copy.deepcopy(board)
        if state['player'] == 0:
            return state

        # Swap players 0 and 1
        for key in ['dice', 'scores', 'used', 'bid_score']:
            state[f'{key}0'], state[f'{key}1'] = state[f'{key}1'], state[f'{key}0']

        if state['pending'] is not None:
            state['pending'] = {
                'player': 1 - state['pending']['player'],
                'choice': state['pending']['choice'],
                'bid_amount': state['pending']['bid_amount']
            }

        if state['pending_scoring'] is not None:
            state['pending_scoring'] = 1 - state['pending_scoring']

        state['player'] = 0
        return state

    def getSymmetries(self, board: Dict, pi: np.ndarray) -> List[Tuple[Dict, np.ndarray]]:
        """Return symmetrical forms (none for yacht game)."""
        return [(board, pi)]

    def stringRepresentation(self, board: Dict) -> str:
        """Return unique string representation for hashing."""
        vec = self._board_to_vector(board)
        return ''.join(map(lambda x: format(int(x * 1000), '03d'), vec))

    def _calculate_category_score(self, cat: int, dice: List[int]) -> int:
        """Calculate score for given category and dice."""
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
        if cat == 6:
            return sum(dice) * 1000
        if cat == 7:
            ok = any(dice.count(i) >= 4 for i in range(1, 7))
            return sum(dice) * 1000 if ok else 0
        if cat == 8:
            pair = triple = False
            for i in range(1, 7):
                cnt = dice.count(i)
                if cnt == 2 or cnt == 5:
                    pair = True
                if cnt == 3 or cnt == 5:
                    triple = True
            return sum(dice) * 1000 if pair and triple else 0
        if cat == 9:
            e = [dice.count(i) > 0 for i in range(1, 7)]
            ok = (e[0] and e[1] and e[2] and e[3]) or (e[1] and e[2]
                                                       and e[3] and e[4]) or (e[2] and e[3] and e[4] and e[5])
            return 15000 if ok else 0
        if cat == 10:
            e = [dice.count(i) > 0 for i in range(1, 7)]
            ok = (e[0] and e[1] and e[2] and e[3] and e[4]) or (
                e[1] and e[2] and e[3] and e[4] and e[5])
            return 30000 if ok else 0
        if cat == 11:
            ok = any(dice.count(i) == 5 for i in range(1, 7))
            return 50000 if ok else 0
        return 0

    def _board_to_vector(self, board: dict) -> np.ndarray:
        """Convert board to neural network input vector."""
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

        # Player dice (10 positions × 7 flags each)
        for player in ['dice0', 'dice1']:
            for pos in range(10):
                if pos < len(board[player]):
                    val = board[player][pos]
                    vec[idx + val - 1] = 1.0
                else:
                    vec[idx + 6] = 1.0
                idx += 7

        # Bundles (2 bundles × 5 dice × 7 flags)
        for bundle in ['bundle_a', 'bundle_b']:
            for pos in range(5):
                if pos < len(board[bundle]):
                    val = board[bundle][pos]
                    vec[idx + val - 1] = 1.0
                else:
                    vec[idx + 6] = 1.0
                idx += 7

        # Used categories (2 players × 12 categories)
        for player in ['used0', 'used1']:
            for used in board[player]:
                vec[idx] = 1.0 if used else 0.0
                idx += 1

        # Scores (2 players × 12 categories, scaled)
        for player in ['scores0', 'scores1']:
            for s in board[player]:
                vec[idx] = s / 100000.0
                idx += 1

        # Bid scores (2 players, scaled)
        vec[idx] = board['bid_score0'] / 100000.0
        idx += 1
        vec[idx] = board['bid_score1'] / 100000.0
        idx += 1

        # Pending information (13 bits total)
        if board['pending'] is not None:
            vec[idx] = 1.0
            idx += 1
            grp = board['pending']['choice']
            vec[idx + grp] = 1.0
            idx += 2
            try:
                bidx = self.bid_levels.index(
                    board['pending']['bid_amount'] // 1000)
            except ValueError:
                bidx = 0
            for i in range(len(self.bid_levels)):
                vec[idx + i] = 1.0 if i == bidx else 0.0
            idx += len(self.bid_levels)
            vec[idx] = 1.0 if board['pending']['player'] != 0 else 0.0
            idx += 1
        else:
            vec[idx] = 0.0
            idx += 1
            idx += 2
            idx += len(self.bid_levels)
            vec[idx] = 0.0
            idx += 1

        # Scoring step flag
        vec[idx] = 1.0 if board['pending_scoring'] is not None else 0.0
        idx += 1

        return vec


# ========================== NEURAL NETWORK ==========================

class Net(nn.Module):
    """Neural network for yacht game."""

    def __init__(self, game: Game):
        super().__init__()
        self.board_size = game.getBoardSize()
        self.action_size = game.getActionSize()
        input_dim = self.board_size[0] * \
            self.board_size[1] * self.board_size[2]
        hidden_dim = 256

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, self.action_size)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy, value.squeeze(-1)


class NNetWrapper:
    """Neural network wrapper for yacht game."""

    def __init__(self, game: Game) -> None:
        self.nnet = Net(game)
        self.game = game
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.nnet.to(self.device)

    def predict(self, board: Dict) -> Tuple[np.ndarray, float]:
        """Predict policy and value for given board state."""
        self.nnet.eval()
        with torch.no_grad():
            board_vec = self.game._board_to_vector(board)
            board_tensor = torch.tensor(
                board_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            policy, value = self.nnet(board_tensor)
            policy = torch.softmax(policy, dim=1).cpu().numpy()[0]
            value = value.item()
        return policy, value

    def load_checkpoint(self, folder: str, filename: str) -> None:
        """Load network parameters from file."""
        filepath = os.path.join(folder, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        self.nnet.load_state_dict(checkpoint['state_dict'])


# ========================== MCTS IMPLEMENTATION ==========================

class MCTS:
    """Monte Carlo Tree Search for yacht game."""

    def __init__(self, game: Game, nnet: NNetWrapper, cpuct: float = 1.0) -> None:
        self.game = game
        self.nnet = nnet
        self.cpuct = cpuct
        self.Qsa: Dict[Tuple[str, int], float] = {}
        self.Nsa: Dict[Tuple[str, int], int] = {}
        self.Ns: Dict[str, int] = {}
        self.Ps: Dict[str, np.ndarray] = {}
        self.Es: Dict[str, float] = {}
        self.Vs: Dict[str, np.ndarray] = {}

    def get_action_prob(self, board: Dict, player: int, temp: float = 1.0, num_sims: int = 50) -> np.ndarray:
        """Get action probabilities from MCTS."""
        for _ in range(num_sims):
            self.search(board, player)

        s = self.game.stringRepresentation(board)
        counts = np.array([self.Nsa.get((s, a), 0) for a in range(
            self.game.getActionSize())], dtype=np.float32)

        if temp == 0:
            best_actions = np.argwhere(counts == np.max(counts)).flatten()
            probs = np.zeros_like(counts)
            probs[random.choice(best_actions)] = 1.0
            return probs

        counts = counts ** (1.0 / temp)
        if counts.sum() == 0:
            counts = np.ones_like(counts)
        probs = counts / counts.sum()
        return probs

    def search(self, board: Dict, player: int) -> float:
        """MCTS search simulation."""
        s = self.game.stringRepresentation(board)

        if s not in self.Es:
            self.Es[s] = self.game.getGameEnded(board, player)
        if self.Es[s] != 0:
            return -self.Es[s]

        if s not in self.Ps:
            policy, v = self.nnet.predict(board)
            valid_moves = self.game.getValidMoves(board, player)
            policy = policy * valid_moves
            sum_p = policy.sum()
            if sum_p > 0:
                policy /= sum_p
            else:
                policy = valid_moves / np.maximum(valid_moves.sum(), 1)

            self.Ps[s] = policy
            self.Vs[s] = valid_moves
            self.Ns[s] = 0
            return -v

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
        next_board, next_player = self.game.getNextState(board, player, a)
        next_board = self.game.getCanonicalForm(next_board, next_player)
        v = self.search(next_board, next_player)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] *
                                self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v


# ========================== UTILITY FUNCTIONS ==========================

def nearest_bid_index(bid_levels: List[int], amount: int) -> int:
    """Find closest bid level index."""
    target = amount // 1000
    best_idx = 0
    best_diff = abs(bid_levels[0] - target)
    for i, level in enumerate(bid_levels):
        diff = abs(level - target)
        if diff < best_diff:
            best_diff = diff
            best_idx = i
    return best_idx


def find_subset_index(dice_pool: List[int], selected: List[int], subsets: List[Tuple[int, ...]]) -> int:
    """Find subset index for selected dice."""
    n = len(dice_pool)
    selected_sorted = sorted(selected)

    for idx, subset in enumerate(subsets):
        if max(subset) >= n:
            continue
        vals = [dice_pool[i] for i in subset]
        if sorted(vals) == selected_sorted:
            return idx
    return -1


# Category name mapping
_CATEGORY_MAP = {
    'ONE': 0, 'TWO': 1, 'THREE': 2, 'FOUR': 3, 'FIVE': 4, 'SIX': 5,
    'CHOICE': 6, 'FOUR_OF_A_KIND': 7, 'FULL_HOUSE': 8,
    'SMALL_STRAIGHT': 9, 'LARGE_STRAIGHT': 10, 'YACHT': 11,
}


# ========================== MAIN AGENT ==========================

def main() -> None:
    """Main agent loop."""
    # Initialize game and neural network
    game = Game()
    nnet = NNetWrapper(game)

    # Load trained model from data.bin in same directory as this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, 'data.bin')

    try:
        nnet.load_checkpoint(script_dir, 'data.bin')
    except Exception as e:
        # Fallback: try other common names
        try:
            nnet.load_checkpoint(script_dir, 'best.pth.tar')
        except Exception:
            print(
                f"Warning: Could not load trained model from {model_path}", file=sys.stderr)

    # Initialize MCTS
    mcts = MCTS(game, nnet, cpuct=1.0)

    # Initialize game state
    board = game.getInitBoard()
    player = 1
    board = game.getCanonicalForm(board, player)

    # Main I/O loop
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
                probs = mcts.get_action_prob(
                    board, player, temp=0, num_sims=50)
                valid = game.getValidMoves(board, player)
                probs = probs * valid

                if probs.sum() == 0:
                    # Fallback to random valid bidding action
                    actions = np.where(valid[:game.num_bid_actions] == 1)[0]
                    action = int(random.choice(actions))
                else:
                    action = int(np.argmax(probs))

                # Decode action
                bundle_choice = action // len(game.bid_levels)
                bid_idx = action % len(game.bid_levels)
                bid_amount = game.bid_levels[bid_idx] * 1000

                # Update internal state
                board, player = game.getNextState(board, player, action)
                board = game.getCanonicalForm(board, player)

                # Output bid
                group_char = 'A' if bundle_choice == 0 else 'B'
                print(f'BID {group_char} {bid_amount}')
                sys.stdout.flush()
                continue

            if command == 'GET':
                # Bidding outcome notification
                my_group_char, opp_group_char, opp_amount_str = parts[1], parts[2], parts[3]
                opp_amount = int(opp_amount_str)

                # Determine opponent's action
                opp_choice = 0 if opp_group_char == 'A' else 1
                opp_bid_idx = nearest_bid_index(game.bid_levels, opp_amount)
                opponent_action = opp_choice * \
                    len(game.bid_levels) + opp_bid_idx

                # Update state with opponent's action
                board, player = game.getNextState(
                    board, player, opponent_action)
                board = game.getCanonicalForm(board, player)
                continue

            if command == 'SCORE':
                # Our turn to select scoring action
                probs = mcts.get_action_prob(
                    board, player, temp=0, num_sims=50)
                valid = game.getValidMoves(board, player)
                probs = probs * valid

                if probs.sum() == 0:
                    # Fallback: choose first valid scoring action
                    scoring_actions = np.where(valid == 1)[0]
                    scoring_actions = [
                        a for a in scoring_actions if a >= game.num_bid_actions]
                    if not scoring_actions:
                        # Emergency fallback - this should not happen
                        action = game.num_bid_actions
                    else:
                        action = int(scoring_actions[0])
                else:
                    # Choose action with highest probability among valid ones
                    valid_actions = np.where(valid == 1)[0]
                    valid_probs = probs[valid_actions]
                    if len(valid_probs) > 0:
                        best_idx = np.argmax(valid_probs)
                        action = valid_actions[best_idx]
                    else:
                        action = int(np.argmax(probs))

                # Decode scoring action
                scoring_action = action - game.num_bid_actions
                cat = scoring_action // game.num_dice_subsets
                subset_idx = scoring_action % game.num_dice_subsets
                subset = game.subsets[subset_idx]

                # Get current dice pool before state update
                dice_pool = board['dice0']

                # Validate and adjust subset to match dice pool
                if max(subset) >= len(dice_pool):
                    # Find a valid subset for current pool size
                    valid_subsets = [
                        s for s in game.subsets if max(s) < len(dice_pool)]
                    if valid_subsets:
                        subset = valid_subsets[0]
                        # Recalculate action to match the new subset
                        new_subset_idx = game.subset_to_index[subset]
                        action = game.num_bid_actions + cat * game.num_dice_subsets + new_subset_idx
                    else:
                        # Pool has fewer than 5 dice, use all available
                        subset = tuple(range(len(dice_pool)))
                        # This case should only happen with < 5 dice total
                        if subset in game.subset_to_index:
                            new_subset_idx = game.subset_to_index[subset]
                            action = game.num_bid_actions + cat * game.num_dice_subsets + new_subset_idx

                # Get the dice values that will actually be used
                chosen_dice_vals = [dice_pool[i] for i in subset]

                # Update state with corrected action
                board, player = game.getNextState(board, player, action)
                board = game.getCanonicalForm(board, player)

                # Output scoring decision
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

                # Find corresponding action
                opp_dice_pool = board['dice1']
                subset_idx = find_subset_index(
                    opp_dice_pool, selected_dice, game.subsets)
                if subset_idx < 0:
                    subset_idx = 0  # Fallback

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
