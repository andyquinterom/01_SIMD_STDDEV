"""Baseline 3: NumPy, `np.std` (vectorized C code).

Usage: uv run 3_numpy.py [N]      (default N = 1,000,000,000)
"""

import sys
import time

import numpy as np

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000_000
RUNS = 10

rng = np.random.default_rng()
print(f"Generating {N:,} random observations (float64)...")
t0 = time.perf_counter()
x = rng.random(N, dtype=np.float64)
print(f"Generation: {time.perf_counter() - t0:.3f} s")

times = []
for i in range(RUNS):
    t = time.perf_counter()
    std = np.std(x)
    times.append(time.perf_counter() - t)
    print(f"Run {i}: {times[-1]:.3f} s")

times.sort()
print(f"Standard deviation: {std:.6f}")
print(f"Median: {times[RUNS // 2]:.3f} s")
