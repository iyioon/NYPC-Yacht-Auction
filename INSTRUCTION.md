---
created: 2025-08-08 17:43
modified: 2025-08-08 18:03
---

# Game Rules (Concise, Model-Ready)

## Overview

- Two-player game of **13 rounds**.
- Each round has a **Bidding** phase and a **Scoring** phase.
  - **Round 1:** Bidding only (no Scoring).
  - **Round 13:** Scoring only (no Bidding).

## Dice & Randomness

- In each Bidding phase (when it exists), roll two independent 5-die bundles: **Bundle A** and **Bundle B**.
- Each die is fair and independent; faces are integers **1–6** with equal probability.
- Any required random tie-break is **uniform** between the tied players and **independent** across occurrences.

## Bidding Phase

1. After the A(5) and B(5) rolls are revealed, each player secretly declares:
   - a **target bundle** `g ∈ {A, B}`, and
   - a **bid** `x`, an integer **0–100000** (inclusive).
2. Assignment rules:
   - If players target **different** bundles: each receives their chosen bundle.
   - If players target the **same** bundle: the **higher bid** receives that bundle; the other receives the remaining bundle.
   - If **both target and bid** are identical: select the bundle’s recipient **at random** (uniform); the other receives the remaining bundle.
3. Bidding score effect **per player**:
   - If you **receive your targeted bundle**, **subtract** your bid `x` from your total score.
   - If you **do not** receive your targeted bundle, **add** your bid `x` to your total score.

## Inter-round Dice Carryover (to Match “10 dice” wording)

- After Bidding, each player holds **5 dice**.
- In rounds that include Scoring (Rounds 2–12), a player will have **10 dice total**: the **5 carried over** from the previous round **+** the **5 newly obtained** from the current Bidding.
- In **Round 13** (no Bidding), a player has only the **5 carried over**.
- In each Scoring phase, the **5 dice you choose to score are consumed**; the **unscored 5** are carried to the next round (if any).

## Scoring Phase

- Choose **exactly 5 dice** from your currently held dice (10 dice in Rounds 2–12, **5 dice in Round 13**).
- Choose **one unused scoring category**. **Each category may be used at most once per player per game.**
- Score according to **one** of the rules below.

### Basic Categories (faces 1–6)

For each chosen category, score:
- `ONE`: (sum of dice showing **1**) × **1000**
- `TWO`: (sum of dice showing **2**) × **1000**
- `THREE`: (sum of dice showing **3**) × **1000**
- `FOUR`: (sum of dice showing **4**) × **1000**
- `FIVE`: (sum of dice showing **5**) × **1000**
- `SIX`: (sum of dice showing **6**) × **1000**

**Basic-section bonus:** If the **total** from the six Basic categories ≥ **63000**, add **+35000** points.

### Combination Categories

Evaluate only the **5 dice you selected** this turn:
- `CHOICE`: (sum of all 5 dice) × **1000**
- `FOUR_OF_A_KIND`: if at least four dice show the same face, score (sum of all 5 dice) × **1000**; otherwise **0**
- `FULL_HOUSE`: if a (3-of-a-kind) **and** a (pair) are both present, score (sum of all 5 dice) × **1000**; otherwise **0**
- `SMALL_STRAIGHT`: if the 5 dice **contain any 4-die run** among {1-2-3-4, 2-3-4-5, 3-4-5-6}, score **15000**; otherwise **0**
- `LARGE_STRAIGHT`: if the 5 dice form **1-2-3-4-5** or **2-3-4-5-6** (all five in sequence), score **30000**; otherwise **0**
- `YACHT`: if all five dice show the **same** face, score **50000**; otherwise **0**

## Final Score & Winner

- A player’s final total = (**sum of all Basic categories** + **Basic bonus if any**) + (**sum of all Combination categories**) + (**net bidding adjustments** across the game).
- Higher total wins; equal totals result in a **draw**.

---

# I/O Protocol (for Programmatic agents)

**General:** One command per line. Outputs must end with a newline and be **flushed**. Missing a required response within the time limit yields **TLE** and a **forfeit** for that game.

| Command | Input format                                  | Your required output (deadline)                                                                                          |
|--------:|-----------------------------------------------|---------------------------------------------------------------------------------------------------------------------------|
| READY   | `READY`                                       | Print `OK` (**within 3.0 s**)                                                                                             |
| ROLL    | `ROLL a1a2a3a4a5 b1b2b3b4b5`                  | Print `BID g x` where `g ∈ {A,B}`, `x ∈ [0,100000]` (**within 0.5 s**)                                                   |
| GET     | `GET g g0 x0`                                 | Notification: you received bundle `g`; opponent targeted `g0` with bid `x0`. **No required immediate output.**           |
| SCORE   | `SCORE`                                       | Print `PUT c d1d2d3d4d5` (**within 0.5 s**), where `c` is an **unused** category and `d1..d5` are the five dice you pick |
| SET     | `SET c d1d2d3d4d5`                            | Notification: opponent used category `c` with dice `d1..d5`. **No required immediate output.**                           |
| FINISH  | `FINISH`                                      | Terminate cleanly **immediately**. **No output.**                                                                         |

**Notes for I/O fields**
- `a1..a5`, `b1..b5` are the pips (1–6) of the rolled dice in bundles A and B.
- `d1..d5` are the pips of the **five dice you select** for this scoring turn (must be among dice you currently hold).
- Time limits are **hard**; respond within the stated windows.
