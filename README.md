# NYPC Yacht Auction AI Agent

An AI agent implementation for the NYPC Yacht Auction game using reinforcement learning techniques based on the AlphaZero algorithm. This project creates an intelligent agent capable of playing the yacht-auction dice game through self-play training and Monte Carlo Tree Search.

## Game Overview

The yacht auction game is a two-player strategic dice game consisting of 13 rounds with bidding and scoring phases:

- **Bidding Phase**: Players bid on dice bundles (A or B) with strategic risk-reward mechanics
- **Scoring Phase**: Players choose dice to score in various categories (basic face values, combinations like Full House, Straight, Yacht)
- **Victory Condition**: Highest total score after 13 rounds wins

For complete game rules, see [`INSTRUCTION.md`](INSTRUCTION.md).

## Project Structure

### Core Framework Files
- `Game.py` - Abstract game interface that will be implemented for yacht auction
- `NeuralNet.py` - Neural network interface for the AI agent
- `MCTS.py` - Monte Carlo Tree Search implementation
- `Coach.py` - Training loop for self-play reinforcement learning
- `Arena.py` - Game arena for evaluating different agents
- `pit.py` - Interface for human vs AI gameplay
- `main.py` - Main training script
- `utils.py` - Utility functions

### Game-Specific Files (To Be Implemented)
- `yacht/` - Directory for yacht auction game implementation
  - `YachtGame.py` - Game logic and rules
  - `YachtPlayers.py` - Human and AI player implementations
  - `YachtNNet.py` - Neural network architecture

## Getting Started

### Prerequisites
Install required dependencies:
```bash
pip install -r requirements.txt
```

### Training an AI Agent
1. Implement the yacht auction game logic in the `yacht/` directory
2. Configure training parameters in `main.py`
3. Start training:
```bash
python main.py
```

### Playing Against the AI
After training, use the pit interface to play against your trained agent:
```bash
python pit.py
```

## Implementation Status

🔄 **In Progress**: 
- Game logic implementation
- Neural network architecture design
- Training configuration

📋 **To Do**:
- Complete yacht auction game rules implementation
- Design state representation for neural network
- Implement bidding and scoring strategy evaluation
- Train and evaluate AI agents

## Technical Approach

This implementation uses:
- **Self-Play Training**: The AI learns by playing against itself
- **Monte Carlo Tree Search**: For strategic move evaluation during gameplay
- **Neural Networks**: To evaluate board positions and suggest moves
- **Experience Replay**: To improve learning from historical games

The agent will learn optimal bidding strategies, dice selection for scoring, and long-term game planning through thousands of self-play iterations.

## License

This project maintains the original license. See [`LICENSE`](LICENSE) for details.

## Acknowledgments

Built upon the Alpha Zero General framework originally developed for games like Othello, adapted specifically for the yacht auction game domain.
