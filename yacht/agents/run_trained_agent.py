#!/usr/bin/env python3
"""
Runner script for the trained MCTS agent.
This script loads the trained neural network and runs the MCTS agent.
"""

import sys
import os

# Add the yacht package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import the MCTS agent
from yacht.agents.agent_mcts import main

if __name__ == '__main__':
    main()
