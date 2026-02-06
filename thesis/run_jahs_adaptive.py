#!/usr/bin/env python3
"""
Root execution script for JAHS Adaptive Benchmark.
Running from here allows proper package resolution.
"""

import sys
import os

# Add thesis root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Delegate to the inner script logic
# We import the module to trigger execution if we move the main logic to a function,
# or we just re-implement the runner here importing from the subfolder.

from alba_framework_potential.benchmark_jahs_adaptive import main

if __name__ == "__main__":
    main()
