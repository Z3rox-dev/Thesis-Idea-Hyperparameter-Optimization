#!/usr/bin/env python3
"""
Root execution script for Real Tabular Benchmark.
"""
import sys
import os

# Delegate to the inner script logic
from alba_framework_potential.benchmark_tabular_real import run_benchmark

if __name__ == "__main__":
    run_benchmark(evals=30, seeds=3)
