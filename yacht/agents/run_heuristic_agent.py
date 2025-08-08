#!/usr/bin/env python3
"""
Script to run the heuristic baseline agent for evaluation.
"""

import sys
import os

# Add the project root to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from yacht.agents.agent import main

if __name__ == '__main__':
    main()
