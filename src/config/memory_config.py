# Memory Optimization Config
import os
import pandas as pd
import numpy as np

# Environment variables for performance
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['NUMEXPR_MAX_THREADS'] = '8'

# Pandas optimizations
pd.options.mode.copy_on_write = True
pd.options.compute.use_numba = True

# LightGBM config
LIGHTGBM_CONFIG = {
    'max_bin': 255,
    'num_threads': 8,
    'device_type': 'cpu',
    'verbose': -1
}

# Memory thresholds
MEMORY_THRESHOLDS = {
    'chunk_size': 100000,  # Process 100k rows at a time
    'max_memory_gb': 8,    # Maximum memory usage
    'sample_size': 50000   # Sample size for large files
}
