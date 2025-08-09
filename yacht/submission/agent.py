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


def _encode_state(game_state, round_no, phase, rollA, rollB, p1_bid, p2_bid, current_player):
    """Encode game state into neural network input vector (59 features)"""
    # Determine which player's perspective
    if current_player == 1:
        me, opp = game_state.my_state, game_state.opp_state
    else:
        me, opp = game_state.opp_state, game_state.my_state

    # Encode features
    features = []

    # Round/phase info (3 features)
    features.append(round_no / 13.0)  # normalize round
    features.append(float(phase))     # 0 for bid, 1 for score
    features.append(float(current_player))  # 1 or -1

    # My dice (10 features) - pad with -1 if fewer than 10
    features.extend(_pad_scale(me.dice, 10))

    # Opponent dice count approximation (10 features) - we don't know exact dice
    opp_dice_count = len(opp.dice)
    opp_dice_approx = [3.5] * min(opp_dice_count, 10) + \
        [0] * max(0, 10 - opp_dice_count)
    features.extend(_pad_scale(opp_dice_approx, 10))

    # Roll A and B (10 features)
    features.extend(_pad_scale(rollA, 5))
    features.extend(_pad_scale(rollB, 5))

    # My used categories mask (12 features)
    for i in range(NUM_CATEGORIES):
        features.append(1.0 if me.rule_score[i] is not None else 0.0)

    # Opponent used categories mask (12 features)
    for i in range(NUM_CATEGORIES):
        features.append(1.0 if opp.rule_score[i] is not None else 0.0)

    # Bid info (2 features)
    my_bid_amount = (p1_bid[1] if current_player ==
                     1 else p2_bid[1]) / 100000.0 if phase == 0 else 0.0
    opp_bid_amount = (p2_bid[1] if current_player ==
                      1 else p1_bid[1]) / 100000.0 if phase == 0 else 0.0
    features.append(my_bid_amount)
    features.append(opp_bid_amount)

    return np.array(features, dtype=np.float32)


def _decode_action(action_idx, game_state, round_no, phase):
    """Decode action index back to game move"""
    if phase == 0:  # Bidding phase
        if action_idx >= NUM_BID_ACTIONS:
            return None  # Invalid action

        target = action_idx // 101  # 0 for A, 1 for B
        amount = (action_idx % 101) * 1000  # 0 to 100000 in steps of 1000
        group = 'A' if target == 0 else 'B'
        return Bid(group, amount)

    else:  # Scoring phase
        scoring_action = action_idx - NUM_BID_ACTIONS
        if scoring_action < 0:
            return None  # Invalid action

        category = scoring_action // NUM_COMB
        comb_idx = scoring_action % NUM_COMB

        if category >= NUM_CATEGORIES or comb_idx >= len(COMB_5_OF_10):
            return None  # Invalid action

        # Check if category is already used
        if game_state.my_state.rule_score[category] is not None:
            return None  # Category already used

        # Get dice combination
        dice_indices = COMB_5_OF_10[comb_idx]
        available_dice = game_state.my_state.dice

        # Check if we have enough dice
        if len(available_dice) < 5 or max(dice_indices) >= len(available_dice):
            return None  # Not enough dice

        selected_dice = [available_dice[i] for i in dice_indices]
        return DicePut(DiceRule(category), selected_dice)


class AIPlayer:
    """Neural network-based AI player"""

    def __init__(self, model_path="data.bin"):
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
            # Encode state
            state_vec = _encode_state(
                game_state, round_no, phase, rollA, rollB, p1_bid, p2_bid, current_player)

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
                # Select action with highest probability among valid actions
                best_action_idx = max(valid_actions, key=lambda x: x[1])[0]
                return _decode_action(best_action_idx, game_state, round_no, phase)

        except Exception as e:
            print(f"AI prediction error: {e}", file=sys.stderr)

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

        # Try AI first
        ai_move = self.ai_player.get_move(
            self, self.round_no, self.phase, self.rollA, self.rollB,
            self.p1_bid, self.p2_bid, 1  # Assume we are player 1
        )

        if ai_move and isinstance(ai_move, Bid):
            return ai_move

        # Fallback to rule-based strategy
        sum_a = sum(dice_a)
        sum_b = sum(dice_b)
        group = "A" if sum_a > sum_b else "B"

        # (내 현재 점수 - 상대 현재 점수) / 10을 0이상 100000이하로 잘라서 배팅
        amount = (
            self.my_state.get_total_score() - self.opp_state.get_total_score()
        ) // 10
        amount = max(0, min(100000, amount))

        return Bid(group, amount)

    # ============================================================================
    # 주어진 주사위에 대해 사용할 규칙과 주사위를 정하는 함수
    # 사용할 규칙과 사용할 주사위의 목록을 pair로 묶어서 반환
    # ============================================================================
    def calculate_put(self) -> DicePut:
        self.phase = 1  # Scoring phase

        # Try AI first
        ai_move = self.ai_player.get_move(
            self, self.round_no, self.phase, self.rollA, self.rollB,
            self.p1_bid, self.p2_bid, 1  # Assume we are player 1
        )

        if ai_move and isinstance(ai_move, DicePut):
            return ai_move

        # Fallback to rule-based strategy
        rule = next(
            i for i, score in enumerate(self.my_state.rule_score) if score is None
        )
        dice = self.my_state.dice[:5]
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

        # 그룹에 따라 주사위 분배
        if my_group == "A":
            self.my_state.add_dice(dice_a)
            self.opp_state.add_dice(dice_b)
        else:
            self.my_state.add_dice(dice_b)
            self.opp_state.add_dice(dice_a)

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

    def update_set(self, put: DicePut):
        """상대가 주사위를 배치한 결과 반영"""
        self.opp_state.use_dice(put)


# 팀의 현재 상태를 관리하는 클래스
class GameState:
    def __init__(self):
        self.dice = []  # 현재 보유한 주사위 목록
        self.rule_score: List[Optional[int]] = [
            None
        ] * 12  # 각 규칙별 획득 점수 (사용하지 않았다면 None)
        self.bid_score = 0  # 입찰로 얻거나 잃은 총 점수

    def get_total_score(self) -> int:
        """현재까지 획득한 총 점수 계산 (상단/하단 점수 + 보너스 + 입찰 점수)"""
        basic = bonus = combination = 0

        # 기본 점수 규칙 계산 (ONE ~ SIX)
        basic = sum(
            score for score in self.rule_score[0:6] if score is not None)
        bonus = 35000 if basic >= 63000 else 0
        combination = sum(
            score for score in self.rule_score[6:12] if score is not None)

        return basic + bonus + combination + self.bid_score

    def bid(self, is_successful: bool, amount: int):
        """입찰 결과에 따른 점수 반영"""
        if is_successful:
            self.bid_score -= amount  # 성공시 베팅 금액만큼 점수 차감
        else:
            self.bid_score += amount  # 실패시 베팅 금액만큼 점수 획득

    def add_dice(self, new_dice: List[int]):
        """새로운 주사위들을 보유 목록에 추가"""
        self.dice.extend(new_dice)

    def use_dice(self, put: DicePut):
        """주사위를 사용하여 특정 규칙에 배치"""
        # 이미 사용한 규칙인지 확인
        assert (
            put.rule is not None and self.rule_score[put.rule.value] is None
        ), "Rule already used"

        for d in put.dice:
            # 주사위 목록에 있는 주사위 제거
            self.dice.remove(d)

        # 해당 규칙의 점수 계산 및 저장
        assert put.rule is not None
        self.rule_score[put.rule.value] = self.calculate_score(put)

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
                # 주사위 굴리기 결과 받기
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
                continue

            if command == "SCORE":
                # 주사위 골라서 배치하기
                put = game.calculate_put()
                game.update_put(put)
                assert put.rule is not None
                print(f"PUT {put.rule.name} {''.join(map(str, put.dice))}")
                continue

            if command == "SET":
                # 상대의 주사위 배치
                rule, str_dice = args
                dice = [int(c) for c in str_dice]
                game.update_set(DicePut(DiceRule[rule], dice))
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
