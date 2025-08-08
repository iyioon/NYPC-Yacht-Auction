#!/usr/bin/env python3
"""
Runner script for the heuristic agent.
This script runs the rule-based heuristic agent as a baseline.
"""

import sys
import os

# Add the yacht package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import the heuristic agent
from yacht.agents.agent import main

if __name__ == '__main__':
    main()
