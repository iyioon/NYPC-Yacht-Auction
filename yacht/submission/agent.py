import os
import sys
import pickle
from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Neural Network Architecture (copied from YachtNNet.py)
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln1(F.silu(self.fc1(x)))
        h = self.dropout(h)
        h = self.ln2(F.silu(self.fc2(h)))
        return x + h


class YachtNNet(nn.Module):
    def __init__(self, input_len=59, action_size=3226, hidden=512, nblocks=8, dropout=0.2):
        super().__init__()
        self.input_len = input_len
        self.action_size = action_size

        self.inp = nn.Sequential(
            nn.Linear(input_len, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden, dropout) for _ in range(nblocks)])

        # Policy head
        self.pi_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, action_size),
        )
        # Value head
        self.v_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.inp(x)
        for block in self.blocks:
            x = block(x)
        pi = self.pi_head(x)
        v = self.v_head(x)
        return pi, v


# Constants from YachtGame
NUM_CATEGORIES = 12
NUM_BID_ACTIONS = 202  # 2 targets * 101 amounts (0-100000 in steps of 1000)
NUM_COMB = 252  # C(10,5) combinations of selecting 5 dice from 10

# Precomputed combinations for selecting 5 dice from 10
COMB_5_OF_10 = []


def _generate_combinations():
    """Generate all combinations of selecting 5 items from 10"""
    from itertools import combinations
    for comb in combinations(range(10), 5):
        COMB_5_OF_10.append(comb)


_generate_combinations()


# State encoding functions (from YachtGame)
def _scale_die(d):
    """1..6 -> roughly [-0.71, 0.71]; pad -1 -> -1.0"""
    return -1.0 if d < 1 else (d - 3.5) / 3.5


def _pad_scale(dice, n):
    """Pad and scale dice array"""
    arr = np.full(n, -1.0, dtype=np.float32)
    for i, v in enumerate(dice[:n]):
        arr[i] = _scale_die(v)
    return arr


def _mask_bits(mask, n):
    """Convert bitmask to array - matches training exactly"""
    return np.array([(mask >> i) & 1 for i in range(n)], dtype=np.float32)


def _encode_state(game_state, round_no, phase, rollA, rollB, p1_bid, p2_bid, current_player):
    """Encode game state into neural network input vector (59 features) - MATCHES TRAINING EXACTLY"""
    # Always canonical form - current player is always "me" (player 1 perspective)
    me, opp = game_state.my_state, game_state.opp_state

    # Encode features in EXACT order as training
    features = []

    # Round/phase info (3 features) - EXACT match with training
    features.append(round_no / 13.0)                    # normalized round
    # is_bid (1.0 during bidding)
    features.append(1.0 if phase == 0 else 0.0)
    # is_score (1.0 during scoring)
    features.append(1.0 if phase == 1 else 0.0)

    # My 10 dice slots (10 features) - use carry (matches training)
    features.extend(_pad_scale(me.carry, 10))

    # Opponent 10 dice slots (10 features) - use actual opponent dice
    features.extend(_pad_scale(opp.carry, 10))

    # Current rolls (10 features) - EXACT logic from training
    # Only show rolls if bidding phase AND not round 13
    if phase == 0 and round_no != 13:
        features.extend(_pad_scale(rollA, 5))
        features.extend(_pad_scale(rollB, 5))
    else:
        features.extend([-1.0] * 10)  # Pad with -1 when not visible

    # Used categories masks (24 features) - use bitmask like training EXACTLY
    features.extend(_mask_bits(me.used_mask, 12))
    features.extend(_mask_bits(opp.used_mask, 12))

    # Bid scores (2 features) - EXACT scaling like training
    features.append(me.bid_score * 1e-5)
    features.append(opp.bid_score * 1e-5)

    assert len(
        features) == 59, f"Feature vector length mismatch: {len(features)} != 59"
    return np.array(features, dtype=np.float32)


def _decode_action(action_idx, game_state, round_no, phase):
    """Decode action index back to game move - MATCHES TRAINING EXACTLY"""
    if phase == 0:  # Bidding phase
        if action_idx >= NUM_BID_ACTIONS:
            return None  # Invalid action

        # Use EXACT same encoding as training (YachtGame.py)
        tidx = action_idx // 101  # BID_LEVELS = 101
        aidx = action_idx % 101
        target = "A" if tidx == 0 else "B"
        amount = aidx * 1000  # BID_STEP = 1000, amounts 0..100000
        return Bid(target, amount)

    else:  # Scoring phase
        scoring_action = action_idx - NUM_BID_ACTIONS
        if scoring_action < 0:
            return None  # Invalid action

        category = scoring_action // NUM_COMB  # NUM_COMB = 252
        comb_idx = scoring_action % NUM_COMB

        if category >= NUM_CATEGORIES or comb_idx >= len(COMB_5_OF_10):
            return None  # Invalid action

        # Check if category is already used - use training-compatible used_mask
        if (game_state.my_state.used_mask >> category) & 1:
            return None  # Category already used

        # Get dice combination - check against carry (like training)
        dice_indices = COMB_5_OF_10[comb_idx]
        available_dice = game_state.my_state.carry  # Use carry like training

        # Check if we have enough dice and valid indices
        if len(available_dice) < 5 or max(dice_indices) >= len(available_dice):
            return None  # Not enough dice or invalid indices

        selected_dice = [available_dice[i] for i in dice_indices]
        return DicePut(DiceRule(category), selected_dice)


class AIPlayer:
    """Neural network-based AI player"""

    def __init__(self, model_path="/Users/jason/Desktop/NYPC-Yacht-Auction/yacht/submission/data.bin"):
        self.model = None
        self.load_model(model_path)

    def load_model(self, model_path):
        """Load the trained model from file"""
        try:
            if os.path.exists(model_path):
                # Load the saved checkpoint
                device = 'cpu'  # Force CPU for submission
                checkpoint = torch.load(
                    model_path, map_location=device, weights_only=False)

                # Extract model args from checkpoint
                args = checkpoint.get('args', {})
                hidden = args.get('hidden', 512)
                nblocks = args.get('nblocks', 8)
                dropout = args.get('dropout', 0.2)

                # Create and load model
                self.model = YachtNNet(
                    input_len=59,
                    action_size=3226,
                    hidden=hidden,
                    nblocks=nblocks,
                    dropout=dropout
                )
                self.model.load_state_dict(checkpoint['state_dict'])
                self.model.eval()
                print("AI model loaded successfully", file=sys.stderr)
            else:
                print(
                    f"Model file {model_path} not found, using fallback AI", file=sys.stderr)
                self.model = None
        except Exception as e:
            print(
                f"Error loading model: {e}, using fallback AI", file=sys.stderr)
            self.model = None

    def get_move(self, game_state, round_no, phase, rollA, rollB, p1_bid, p2_bid, current_player):
        """Get the best move using the neural network"""
        if self.model is None:
            return None  # Fall back to rule-based

        try:
            print(
                f"# Debug: get_move called with round={round_no}, phase={phase}", file=sys.stderr)

            # Encode state
            state_vec = _encode_state(
                game_state, round_no, phase, rollA, rollB, p1_bid, p2_bid, current_player)

            print(
                f"# Debug: State vector (first 10): {state_vec[:10]}", file=sys.stderr)
            print(f"# Debug: Round {round_no}, Phase {phase}", file=sys.stderr)
            print(
                f"# Debug: My carry dice: {game_state.my_state.carry}", file=sys.stderr)
            print(
                f"# Debug: Opp carry dice: {game_state.opp_state.carry}", file=sys.stderr)
            print(
                f"# Debug: Bid scores: me={game_state.my_state.bid_score}, opp={game_state.opp_state.bid_score}", file=sys.stderr)
            print(
                f"# Debug: State vector round/phase features: {state_vec[:3]}", file=sys.stderr)

            # Get neural network prediction
            with torch.no_grad():
                state_tensor = torch.from_numpy(state_vec).unsqueeze(0).float()
                pi, v = self.model(state_tensor)
                pi = F.softmax(pi, dim=1).cpu().numpy()[0]

            valid_actions = []
            for action_idx in range(len(pi)):
                move = _decode_action(action_idx, game_state, round_no, phase)
                if move is not None:
                    valid_actions.append((action_idx, pi[action_idx]))

            if valid_actions:
                # Sort by probability to see top choices
                valid_actions.sort(key=lambda x: x[1], reverse=True)

                # Debug: show top 5 actions during bidding
                if phase == 0:
                    print(f"# Debug: Top 5 bid actions:", file=sys.stderr)
                    for i, (action_idx, prob) in enumerate(valid_actions[:5]):
                        move = _decode_action(
                            action_idx, game_state, round_no, phase)
                        if isinstance(move, Bid):
                            print(
                                f"# Debug:   {i + 1}. Action {action_idx} -> BID {move.group} {move.amount} (prob: {prob:.6f})", file=sys.stderr)

                # Select action with highest probability among valid actions
                # Already sorted by probability
                best_action_idx = valid_actions[0][0]
                best_move = _decode_action(
                    best_action_idx, game_state, round_no, phase)

                if phase == 0 and isinstance(best_move, Bid):
                    print(
                        f"# Debug: AI chose action {best_action_idx} -> BID {best_move.group} {best_move.amount}", file=sys.stderr)
                elif phase == 1 and isinstance(best_move, DicePut):
                    print(
                        f"# Debug: AI chose action {best_action_idx} -> PUT {best_move.rule.name} {best_move.dice}", file=sys.stderr)

                return best_move

        except Exception as e:
            print(f"AI prediction error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

        return None  # Fall back to rule-based


# 가능한 주사위 규칙들을 나타내는 enum
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


# 입찰 방법을 나타내는 데이터클래스
@dataclass
class Bid:
    group: str  # 입찰 그룹 ('A' 또는 'B')
    amount: int  # 입찰 금액


# 주사위 배치 방법을 나타내는 데이터클래스
@dataclass
class DicePut:
    rule: DiceRule  # 배치 규칙
    dice: List[int]  # 배치할 주사위 목록


# 게임 상태를 관리하는 클래스
class Game:
    def __init__(self):
        self.my_state = GameState()  # 내 팀의 현재 상태
        self.opp_state = GameState()  # 상대 팀의 현재 상태
        self.ai_player = AIPlayer()   # AI player instance
        self.round_no = 1            # Current round number
        self.phase = 0               # Current phase (0=bid, 1=score)
        self.rollA = []              # Current roll A
        self.rollB = []              # Current roll B
        self.p1_bid = ('', 0)        # Player 1 bid (group, amount)
        self.p2_bid = ('', 0)        # Player 2 bid (group, amount)

    # ================================ [필수 구현] ================================
    # ============================================================================
    # 주사위가 주어졌을 때, 어디에 얼마만큼 베팅할지 정하는 함수
    # 입찰할 그룹과 베팅 금액을 pair로 묶어서 반환
    # ============================================================================
    def calculate_bid(self, dice_a: List[int], dice_b: List[int]) -> Bid:
        self.rollA = dice_a
        self.rollB = dice_b
        self.phase = 0  # Bidding phase

        # AI only - no fallback
        ai_move = self.ai_player.get_move(
            self, self.round_no, self.phase, self.rollA, self.rollB,
            self.p1_bid, self.p2_bid, 1  # Assume we are player 1
        )

        if ai_move and isinstance(ai_move, Bid):
            return ai_move

        # If AI fails, make a minimal bid to continue the game
        return Bid("A", 0)

    # ============================================================================
    # 주어진 주사위에 대해 사용할 규칙과 주사위를 정하는 함수
    # 사용할 규칙과 사용할 주사위의 목록을 pair로 묶어서 반환
    # ============================================================================
    def calculate_put(self) -> DicePut:
        self.phase = 1  # Scoring phase

        # AI only - no fallback
        ai_move = self.ai_player.get_move(
            self, self.round_no, self.phase, self.rollA, self.rollB,
            self.p1_bid, self.p2_bid, 1  # Assume we are player 1
        )

        if ai_move and isinstance(ai_move, DicePut):
            return ai_move

        # If AI fails, make a minimal move to continue the game
        # Find first unused category using used_mask
        for i in range(NUM_CATEGORIES):
            if ((self.my_state.used_mask >> i) & 1) == 0:
                rule = i
                break
        else:
            rule = 0  # Fallback if all used (shouldn't happen)

        dice = self.my_state.carry[:5]  # Use carry instead of dice
        return DicePut(DiceRule(rule), dice)

    # ============================== [필수 구현 끝] ==============================

    def update_get(
        self,
        dice_a: List[int],
        dice_b: List[int],
        my_bid: Bid,
        opp_bid: Bid,
        my_group: str,
    ):
        """입찰 결과를 받아서 상태 업데이트"""
        # Store bid information
        self.p1_bid = (my_bid.group, my_bid.amount)
        self.p2_bid = (opp_bid.group, opp_bid.amount)

        # 그룹에 따라 주사위 분배 - I know exactly what opponent got
        if my_group == "A":
            self.my_state.add_dice(dice_a)
            self.opp_state.add_dice(dice_b)  # Opponent got dice_b
        else:
            self.my_state.add_dice(dice_b)
            self.opp_state.add_dice(dice_a)  # Opponent got dice_a

        # 입찰 결과에 따른 점수 반영
        my_bid_ok = my_bid.group == my_group
        self.my_state.bid(my_bid_ok, my_bid.amount)

        opp_group = "B" if my_group == "A" else "A"
        opp_bid_ok = opp_bid.group == opp_group
        self.opp_state.bid(opp_bid_ok, opp_bid.amount)

    def update_put(self, put: DicePut):
        """내가 주사위를 배치한 결과 반영"""
        self.my_state.use_dice(put)

    def update_set(self, put: DicePut):
        """상대가 주사위를 배치한 결과 반영"""
        self.opp_state.use_dice(put)

    def advance_round(self):
        """Advance to next round"""
        self.round_no += 1
        self.p1_bid = ('', 0)
        self.p2_bid = ('', 0)


# 팀의 현재 상태를 관리하는 클래스 - MATCHES TRAINING PlayerState EXACTLY
class GameState:
    def __init__(self):
        self.carry = []  # Dice carried from previous round
        self.used_mask = 0  # 12-bit mask of used categories (matches training)
        # Category scores (matches training)
        self.cat_scores = [0] * NUM_CATEGORIES
        self.bid_score = 0  # Net bidding adjustments (matches training)

    @property
    def rule_score(self):
        """Compatibility property - convert cat_scores to rule_score format for existing code"""
        return [self.cat_scores[i] if (self.used_mask >> i) & 1 else None for i in range(NUM_CATEGORIES)]

    def get_total_score(self) -> int:
        """현재까지 획득한 총 점수 계산 (상단/하단 점수 + 보너스 + 입찰 점수)"""
        basic = sum(self.cat_scores[0:6])
        bonus = 35000 if basic >= 63000 else 0
        combination = sum(self.cat_scores[6:12])
        return basic + bonus + combination + self.bid_score

    def bid(self, is_successful: bool, amount: int):
        """입찰 결과에 따른 점수 반영"""
        if is_successful:
            self.bid_score -= amount  # 성공시 베팅 금액만큼 점수 차감
        else:
            self.bid_score += amount  # 실패시 베팅 금액만큼 점수 획득

    def add_dice(self, new_dice: List[int]):
        """새로운 주사위들을 보유 목록에 추가 - 이제 carry 시스템 사용"""
        self.carry.extend(new_dice)

    def use_dice(self, put: DicePut):
        """주사위를 사용하여 특정 규칙에 배치 - matches training PlayerState exactly"""
        # Check if category is already used
        assert put.rule is not None, "Rule cannot be None"
        cat_idx = put.rule.value
        assert ((self.used_mask >> cat_idx) & 1) == 0, "Rule already used"

        # Remove dice by values (the exact dice specified in put.dice)
        remaining_dice = list(self.carry)

        # Find and remove each die from put.dice
        for target_die in put.dice:
            for i, die in enumerate(remaining_dice):
                if die == target_die:
                    remaining_dice[i] = -1  # Mark as used
                    break

        # Remove marked dice
        self.carry = [die for die in remaining_dice if die != -1]

        # Mark category as used and store score - MATCHES TRAINING
        self.used_mask |= (1 << cat_idx)
        self.cat_scores[cat_idx] = self.calculate_score(put)

    @staticmethod
    def calculate_score(put: DicePut) -> int:
        """규칙에 따른 점수를 계산하는 함수"""
        rule, dice = put.rule, put.dice

        # 기본 규칙 점수 계산 (해당 숫자에 적힌 수의 합 × 1000점)
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
        if rule == DiceRule.CHOICE:  # 주사위에 적힌 모든 수의 합 × 1000점
            return sum(dice) * 1000
        if (
            rule == DiceRule.FOUR_OF_A_KIND
        ):  # 같은 수가 적힌 주사위가 4개 있다면, 주사위에 적힌 모든 수의 합 × 1000점, 아니면 0
            ok = any(dice.count(i) >= 4 for i in range(1, 7))
            return sum(dice) * 1000 if ok else 0
        if (
            rule == DiceRule.FULL_HOUSE
        ):  # 3개의 주사위에 적힌 수가 서로 같고, 다른 2개의 주사위에 적힌 수도 서로 같으면 주사위에 적힌 모든 수의 합 × 1000점, 아닐 경우 0점
            pair = triple = False
            for i in range(1, 7):
                cnt = dice.count(i)
                # 5개 모두 같은 숫자일 때도 인정
                if cnt == 2 or cnt == 5:
                    pair = True
                if cnt == 3 or cnt == 5:
                    triple = True
            return sum(dice) * 1000 if pair and triple else 0
        if (
            rule == DiceRule.SMALL_STRAIGHT
        ):  # 4개의 주사위에 적힌 수가 1234, 2345, 3456중 하나로 연속되어 있을 때, 15000점, 아닐 경우 0점
            e1, e2, e3, e4, e5, e6 = [dice.count(i) > 0 for i in range(1, 7)]
            ok = (
                (e1 and e2 and e3 and e4)
                or (e2 and e3 and e4 and e5)
                or (e3 and e4 and e5 and e6)
            )
            return 15000 if ok else 0
        if (
            rule == DiceRule.LARGE_STRAIGHT
        ):  # 5개의 주사위에 적힌 수가 12345, 23456중 하나로 연속되어 있을 때, 30000점, 아닐 경우 0점
            e1, e2, e3, e4, e5, e6 = [dice.count(i) > 0 for i in range(1, 7)]
            ok = (e1 and e2 and e3 and e4 and e5) or (
                e2 and e3 and e4 and e5 and e6)
            return 30000 if ok else 0
        if (
            rule == DiceRule.YACHT
        ):  # 5개의 주사위에 적힌 수가 모두 같을 때 50000점, 아닐 경우 0점
            ok = any(dice.count(i) == 5 for i in range(1, 7))
            return 50000 if ok else 0

        assert False, "Invalid rule"


def main():
    game = Game()

    # 입찰 라운드에서 나온 주사위들
    dice_a, dice_b = [0] * 5, [0] * 5
    # 내가 마지막으로 한 입찰 정보
    my_bid = Bid("", 0)

    while True:
        try:
            line = input().strip()
            if not line:
                continue

            command, *args = line.split()

            if command == "READY":
                # 게임 시작
                print("OK")
                continue

            if command == "ROLL":
                # 주사위 굴리기 결과 받기 - implies start of new round if we just finished scoring
                str_a, str_b = args
                for i, c in enumerate(str_a):
                    dice_a[i] = int(c)  # 문자를 숫자로 변환
                for i, c in enumerate(str_b):
                    dice_b[i] = int(c)  # 문자를 숫자로 변환
                my_bid = game.calculate_bid(dice_a, dice_b)
                print(f"BID {my_bid.group} {my_bid.amount}")
                continue

            if command == "GET":
                # 주사위 받기
                get_group, opp_group, opp_score = args
                opp_score = int(opp_score)
                game.update_get(
                    dice_a, dice_b, my_bid, Bid(
                        opp_group, opp_score), get_group
                )
                # In round 1, there's no scoring, so advance round here
                if game.round_no == 1:
                    game.advance_round()
                continue

            if command == "SCORE":
                # 주사위 골라서 배치하기
                put = game.calculate_put()
                game.update_put(put)
                assert put.rule is not None
                print(f"PUT {put.rule.name} {''.join(map(str, put.dice))}")
                continue

            if command == "SET":
                # 상대의 주사위 배치 - this means both players have scored this round
                rule, str_dice = args
                dice = [int(c) for c in str_dice]
                game.update_set(DicePut(DiceRule[rule], dice))
                # After both players score, advance to next round (rounds 2-12 only)
                if 2 <= game.round_no <= 12:
                    game.advance_round()
                continue

            if command == "FINISH":
                # 게임 종료
                break

            # 알 수 없는 명령어 처리
            print(f"Invalid command: {command}", file=sys.stderr)
            sys.exit(1)

        except EOFError:
            break


if __name__ == "__main__":
    main()
