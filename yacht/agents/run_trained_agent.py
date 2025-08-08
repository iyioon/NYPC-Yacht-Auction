#!/usr/bin/env python3
"""
Script to run the trained neural network agent for evaluation.
This loads the best checkpoint from training and uses MCTS for decision making.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from yacht.agents.agent_mcts import main

if __name__ == '__main__':
    main()
