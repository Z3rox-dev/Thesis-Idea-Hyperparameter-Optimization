#!/usr/bin/env python3
"""Debug YAHPO query"""
import sys
sys.path.insert(0, '/mnt/workspace/thesis')
sys.path.insert(0, '/mnt/workspace')

import numpy as np
import warnings
warnings.filterwarnings("ignore")

if not hasattr(np, "bool"):
    np.bool = bool

from yahpo_gym import BenchmarkSet, local_config

local_config.init_config()
local_config.set_data_path("/mnt/workspace/data")

bench = BenchmarkSet("rbv2_xgboost")
print(f"Instances: {list(bench.instances)[:5]}")

bench.set_instance(list(bench.instances)[0])
cs = bench.get_opt_space()

print(f"\nConfig space hyperparams:")
for hp in cs.get_hyperparameters():
    print(f"  {hp.name}: {hp}")

# Sample a config
config = cs.sample_configuration()
print(f"\nSampled config: {dict(config)}")

# Query
try:
    res = bench.objective_function(dict(config))
    print(f"Result: {res}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
