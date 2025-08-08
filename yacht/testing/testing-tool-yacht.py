#!/usr/bin/env python3

import argparse
from enum import Enum
import io
import json
import queue
import subprocess
import sys
from dataclasses import dataclass
import threading
from typing import List, Optional, TextIO, Tuple


class AbortError(Exception):
    """
    Exception raised when the game is aborted due to invalid input.
    """


@dataclass
class GameTurn:
    """
    Single bidding round for yacht dice
    - diceA, diceB: list of 5 dices
    - tieBreaker: 0 or 1, which side gets dice in case of same bet.
    """

    diceA: List[int]
    diceB: List[int]
    tieBreaker: int


@dataclass
class Settings:
    """
    Settings for yacht game
    - inputData: list of (<diceA>, <diceB>, tiebreaker) tuple
    - exec1: command to execute first player
    - exec2: command to execute second player
    """

    inputData: List[GameTurn]
    exec1: str
    exec2: str


class DiceRule(Enum):
    """
    Enum for dice rules.
    """

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
    """
    Bid for a single round.
    - group: 'A' or 'B' to bet
    - amount: 0 to 100000
    """

    group: str
    amount: int

    def __str__(self):
        return f"{self.group} {self.amount}"


def parseBid(s: str) -> Bid:
    try:
        group, amount = s.strip().split()
        if group not in ("A", "B"):
            raise ValueError(f"Invalid group in bid: {group}")
        amount = int(amount)
        if not (0 <= amount <= 100000):
            raise ValueError(f"Invalid amount in bid: {amount}")
        return Bid(group=group, amount=amount)
    except Exception:
        raise AbortError("BID PARSE FAILED")


@dataclass
class DicePut:
    """
    DicePut for a single round.
    - rule: DiceRule
    - dice: list of 5 dices
    """

    rule: DiceRule
    dice: List[int]

    def __str__(self):
        return f"{self.rule.name} {''.join(map(str, self.dice))}"


def parseDicePut(s: str) -> DicePut:
    try:
        rule, dice = s.strip().split()
        dice = list(map(int, dice))
        if not len(dice) == 5 or not all(1 <= d <= 6 for d in dice):
            raise ValueError(f"Invalid dice in dice put: {dice}")
        return DicePut(rule=DiceRule[rule], dice=dice)
    except Exception:
        raise AbortError("PUT PARSE FAILED")


class GameState:
    """
    GameState from `sample-code.py`, with some modifications on raising exceptions.
    """

    def __init__(self):
        self.dice = []
        self.ruleScore: List[Optional[int]] = [None] * 12
        self.bidScore = 0

    def getTotalScore(self) -> int:
        basic = bonus = combination = 0

        basic = sum(score for score in self.ruleScore[0:6] if score is not None)
        bonus = 35000 if basic >= 63000 else 0
        combination = sum(score for score in self.ruleScore[6:12] if score is not None)

        return basic + bonus + combination + self.bidScore

    def bid(self, is_successful: bool, amount: int):
        if is_successful:
            self.bidScore -= amount
        else:
            self.bidScore += amount

    def addDice(self, new_dice: List[int]):
        self.dice.extend(new_dice)

    def useDice(self, put: DicePut):
        if put.rule is not None and self.ruleScore[put.rule.value] is not None:
            raise AbortError("RULE ALREADY USED")

        for d in put.dice:
            try:
                self.dice.remove(d)
            except ValueError:
                raise AbortError("NO SUCH DICE")

        self.ruleScore[put.rule.value] = self.calculateScore(put)

    @staticmethod
    def calculateScore(put: DicePut) -> int:
        rule, dice = put.rule, put.dice

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
        if rule == DiceRule.CHOICE:
            return sum(dice) * 1000
        if rule == DiceRule.FOUR_OF_A_KIND:
            ok = any(dice.count(i) >= 4 for i in range(1, 7))
            return sum(dice) * 1000 if ok else 0
        if rule == DiceRule.FULL_HOUSE:
            pair = triple = False
            for i in range(1, 7):
                cnt = dice.count(i)
                if cnt == 2 or cnt == 5:
                    pair = True
                if cnt == 3 or cnt == 5:
                    triple = True
            return sum(dice) * 1000 if pair and triple else 0
        if rule == DiceRule.SMALL_STRAIGHT:
            e1, e2, e3, e4, e5, e6 = [dice.count(i) > 0 for i in range(1, 7)]
            ok = (
                (e1 and e2 and e3 and e4)
                or (e2 and e3 and e4 and e5)
                or (e3 and e4 and e5 and e6)
            )
            return 15000 if ok else 0
        if rule == DiceRule.LARGE_STRAIGHT:
            e1, e2, e3, e4, e5, e6 = [dice.count(i) > 0 for i in range(1, 7)]
            ok = (e1 and e2 and e3 and e4 and e5) or (e2 and e3 and e4 and e5 and e6)
            return 30000 if ok else 0
        if rule == DiceRule.YACHT:
            ok = any(dice.count(i) == 5 for i in range(1, 7))
            return 50000 if ok else 0

        assert False, "Invalid rule"


def runGame(settings: Settings, res: TextIO):
    """
    Run the game from settings, and returns the result as a string.
    """

    def p(x):
        return ["FIRST", "SECOND"][x]

    e1 = json.dumps(f"COMMAND: {settings.exec1}", ensure_ascii=False)
    e2 = json.dumps(f"COMMAND: {settings.exec2}", ensure_ascii=False)
    res.write(f"[{p(0)} {e1}]\n[{p(1)} {e2}]\n")

    f = io.StringIO()
    users = [Player(0, settings.exec1, f), Player(1, settings.exec2, f)]
    result = "*"

    try:
        # Ready phase
        for u in users:
            u.print("READY")
        lines = Player.readAll(users, 3.0)
        for i, line in enumerate(lines):
            if line is None:
                f.write(f"ABORT {p(i)} TLE\n")
                result = f"{str(i)}-{str(1-i)}"
                return
            if line.strip() != "OK":
                f.write(f"ABORT {p(i)} INVALID READY MESSAGE\n")
                result = f"{str(i)}-{str(1-i)}"
                return

        # Initialize user states
        userStates = [GameState(), GameState()]

        for round in range(1, 13 + 1):
            f.write(f"ROUND {round}\n")

            # BID phase
            if round != 13:
                turnInfo = settings.inputData[round - 1]
                for u in users:
                    u.print(
                        f"ROLL {''.join(map(str, turnInfo.diceA))} {''.join(map(str, turnInfo.diceB))}"
                    )
                f.write(
                    f"ROLL {''.join(map(str, turnInfo.diceA))} {''.join(map(str, turnInfo.diceB))}\n"
                )
                lines = Player.readAll(users, 0.5)
                bidInfos: List[Bid] = []
                for i, line in enumerate(lines):
                    if line is None:
                        f.write(f"ABORT {p(i)} TLE\n")
                        result = f"{str(i)}-{str(1-i)}"
                        return
                    try:
                        line = line.strip()
                        if not line.startswith("BID "):
                            raise AbortError("BID PARSE FAILED")
                        bidInfos.append(parseBid(line[len("BID ") :]))
                    except AbortError as e:
                        f.write(f"ABORT {p(i)} {e}\n")
                        result = f"{str(i)}-{str(1-i)}"
                        return

                f.write(f"BID FIRST {bidInfos[0]}\n")
                f.write(f"BID SECOND {bidInfos[1]}\n")

                # Determine winner
                getGroups = [b.group for b in bidInfos]
                if getGroups[0] == getGroups[1]:
                    if bidInfos[0].amount > bidInfos[1].amount or (
                        bidInfos[0].amount == bidInfos[1].amount
                        and turnInfo.tieBreaker == 0
                    ):
                        getGroups[1] = "B" if getGroups[1] == "A" else "A"
                    else:
                        getGroups[0] = "B" if getGroups[0] == "A" else "A"

                for u in range(2):
                    userStates[u].bid(
                        getGroups[u] == bidInfos[u].group, bidInfos[u].amount
                    )

                userStates[0].addDice(
                    turnInfo.diceA if getGroups[0] == "A" else turnInfo.diceB
                )
                users[0].print(f"GET {getGroups[0]} {bidInfos[1]}")
                f.write(f"GET FIRST {getGroups[0]}\n")
                userStates[1].addDice(
                    turnInfo.diceA if getGroups[1] == "A" else turnInfo.diceB
                )
                users[1].print(f"GET {getGroups[1]} {bidInfos[0]}")
                f.write(f"GET SECOND {getGroups[1]}\n")

            # PUT Phase
            if round != 1:
                for u in users:
                    u.print("SCORE")

                lines = Player.readAll(users, 0.5)
                putInfos: List[DicePut] = []
                for i, line in enumerate(lines):
                    if line is None:
                        f.write(f"ABORT {p(i)} TLE\n")
                        result = f"{str(i)}-{str(1-i)}"
                        return
                    try:
                        line = line.strip()
                        if not line.startswith("PUT "):
                            raise AbortError("PUT PARSE FAILED")
                        putInfos.append(parseDicePut(line[len("PUT ") :]))
                    except AbortError as e:
                        f.write(f"ABORT {p(i)} {e}\n")
                        result = f"{str(i)}-{str(1-i)}"
                        return
                for u in range(2):
                    try:
                        userStates[u].useDice(putInfos[u])
                    except AbortError as e:
                        f.write(f"ABORT {p(i)} {e}\n")
                        result = f"{str(i)}-{str(1-i)}"
                        return

                f.write(f"PUT FIRST {putInfos[0]}\n")
                f.write(f"PUT SECOND {putInfos[1]}\n")
                users[0].print(f"SET {putInfos[1]}")
                users[1].print(f"SET {putInfos[0]}")

        # Game is finished
        f.write("FINISH\n")
        score0, score1 = userStates[0].getTotalScore(), userStates[1].getTotalScore()
        f.write(f"SCOREFIRST {score0}\n")
        f.write(f"SCORESECOND {score1}\n")
        if score0 > score1:
            result = f"1-0"
        elif score0 < score1:
            result = f"0-1"
        else:
            result = f"1/2-1/2"

    finally:
        for u in users:
            u.print("FINISH")
            u.join()
        res.write(f'[RESULT "{result}"]\n')
        res.write(f.getvalue())


class Player:
    """
    Process for a player, that supports rw to stdin/stdout/stderr and terminating the process".
    """

    def __init__(self, no: int, exec: str, logStream: TextIO):
        self.name = ["FIRST", "SECOND"][no]
        self.exec = exec
        try:
            self.process = subprocess.Popen(
                self.exec,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                shell=True,
            )
        except Exception as e:
            print(f"Error: failed to start process {self.exec}: {e}")
            sys.exit(1)

        self.reads = queue.Queue()
        self.writes = queue.Queue()
        self.logStream = logStream

        self.stdin_thread = threading.Thread(target=self.__handle_stdin)
        self.stdout_thread = threading.Thread(target=self.__handle_stdout)
        self.stderr_thread = threading.Thread(target=self.__handle_stderr)
        self.stdin_thread.start()
        self.stdout_thread.start()
        self.stderr_thread.start()

    def __handle_stdin(self):
        """
        Write to the player's stdin.
        """
        stdin = self.process.stdin
        assert stdin is not None
        try:
            while True:
                res = self.writes.get()
                if res is None:
                    break
                stdin.write(f"{res}\n")
                stdin.flush()
        finally:
            stdin.close()

    def __handle_stdout(self):
        """
        Read from the player's stdout, and put the result to the queue.
        """
        stdout = self.process.stdout
        assert stdout is not None
        try:
            while True:
                r = stdout.readline()
                if not r:
                    break
                self.reads.put(r)
        except:
            pass
        finally:
            stdout.close()

    def __handle_stderr(self):
        """
        Read from the player's stderr, and redirect to the log file.
        """
        stderr = self.process.stderr
        assert stderr is not None
        try:
            while True:
                r = stderr.readline()
                if not r:
                    break
                self.logStream.write(f"# Debug {self.name}: {r.rstrip()}\n")
        except:
            pass
        finally:
            stderr.close()

    def print(self, message: str):
        """
        Print a message to the player's stdin. Newline is added automatically.
        """
        self.writes.put(message)

    def readline(self, timeout: float) -> Optional[str]:
        """
        Read a line from the player's stdout.
        Return None if timeout.
        """
        try:
            return self.reads.get(timeout=timeout)
        except queue.Empty:
            return None

    @classmethod
    def readAll(cls, selfs: List["Player"], timeout: float) -> List[Optional[str]]:
        """
        Read all lines from the players' stdout.
        Return None if timeout.
        """

        def __readline_thread(
            p: "Player", timeout: float, idx: int, arr: List[Optional[str]]
        ):
            arr[idx] = p.readline(timeout)

        readline_threads = []
        returns: List[Optional[str]] = [None] * len(selfs)
        for i, p in enumerate(selfs):
            readline_threads.append(
                threading.Thread(
                    target=__readline_thread, args=(p, timeout, i, returns)
                )
            )

        for thread in readline_threads:
            thread.start()

        for thread in readline_threads:
            thread.join()

        return returns

    def join(self, timeout: Optional[float] = 1.0):
        """
        Join the player's process.
        If timeout, terminate the process (SIGTERM in POSIX) and wait for it to exit.
        If the process is not terminated within timeout, kill it.
        """
        self.writes.put(None)
        try:
            self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()


def readInput(f: TextIO) -> List[GameTurn]:
    """
    Read input data from file.
    """

    inputData = []
    for _ in range(12):
        try:
            diceA, diceB, tieBreaker = f.readline().split()
            diceA = list(map(int, diceA))
            diceB = list(map(int, diceB))
            tieBreaker = int(tieBreaker)
            if (
                len(diceA) != 5
                or len(diceB) != 5
                or not all(1 <= a <= 6 for a in diceA)
                or not all(1 <= b <= 6 for b in diceB)
            ):
                raise ValueError(
                    "Invalid input file: diceA and diceB must be string of 5 digits from 1 to 6."
                )

            if tieBreaker not in [0, 1]:
                raise ValueError("Tiebreaker should be 0 or 1.")

            inputData.append(GameTurn(diceA, diceB, tieBreaker))
        except Exception as e:
            print(f"Invalid input file: {e}", file=sys.stderr)
            sys.exit(1)

    return inputData


def readSettings() -> Tuple[TextIO, Settings]:
    """
    Read configs from command line arguments or config file.
    Return (logFile, settings) tuple.
    """

    parser = argparse.ArgumentParser(
        prog="testing-tool-yacht",
        description="Testing tool for yacht dice",
        epilog="For detailed information, please see README.md.",
    )

    parser.add_argument("-c", "--config", type=str, help="predefined config file")
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input file",
    )
    parser.add_argument("-l", "--log", type=str, help="Log file")

    parser.add_argument(
        "-s",
        "--stdio",
        nargs="?",
        const=True,
        type=lambda x: True if x is None else x.lower() == "true",
        default=False,
        help="Use stdandard input/output for input and log file",
    )

    parser.add_argument("-a", "--exec1", type=str, help="First player command")
    parser.add_argument("-b", "--exec2", type=str, help="Second player command")

    args = parser.parse_args()

    inputFile = args.input
    logFile = args.log
    exec1 = args.exec1
    exec2 = args.exec2
    stdio = args.stdio

    # Read key=value settings
    if args.config:
        try:
            f = open(args.config, "r")
        except FileNotFoundError:
            parser.print_help()
            print(f"\nError: Config file {args.config} not found.")
            sys.exit(1)

        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = map(str.strip, line.split("=", 1))
                if key == "INPUT":
                    if inputFile is None:
                        inputFile = value
                elif key == "LOG":
                    if logFile is None:
                        logFile = value
                elif key == "EXEC1":
                    if exec1 is None:
                        exec1 = value
                elif key == "EXEC2":
                    if exec2 is None:
                        exec2 = value
                else:
                    parser.print_help()
                    print(f"\nUnknown line: {line}", file=sys.stderr)
                    sys.exit(1)
            else:
                parser.print_help()
                print(f"\nUnknown line: {line}", file=sys.stderr)
                sys.exit(1)

    # Specify file stream
    if not inputFile:
        if not stdio:
            parser.print_help()
            print("\nError: No input file provided.", file=sys.stderr)
            sys.exit(1)
        else:
            f = sys.stdin
    else:
        try:
            f = open(inputFile, "r")
        except:
            parser.print_help()
            print(f"\nError: Input file {inputFile} not found.", file=sys.stderr)
            sys.exit(1)

    # Read input
    inputData = readInput(f)

    if not stdio and not logFile:
        parser.print_help()
        print("\nError: No log output file provided.", file=sys.stderr)
        sys.exit(1)

    if not exec1:
        parser.print_help()
        print("\nError: First player command not specified.", file=sys.stderr)
        sys.exit(1)
    if not exec2:
        parser.print_help()
        print("\nError: Second player command not specified.", file=sys.stderr)
        sys.exit(1)

    if logFile is None:
        logStream = sys.stdout
    else:
        try:
            logStream = open(logFile, "w")
        except:
            parser.print_help()
            print(f"\nError: Log file {logFile} not found.", file=sys.stderr)
            sys.exit(1)

    return (logStream, Settings(inputData, exec1, exec2))


def main():
    logStream, settings = readSettings()
    runGame(settings, logStream)
    logStream.close()


if __name__ == "__main__":
    main()
