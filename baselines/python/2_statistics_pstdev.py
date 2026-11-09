"""Baseline 2: Python standard library, `statistics.pstdev`.

Usage: uv run 2_statistics_pstdev.py [N]      (default N = 1,000,000,000)
"""

import random
import statistics
import sys
import time
from array import array

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000_000
RUNS = 10

print(f"Generating {N:,} random observations (float64)...")
t0 = time.perf_counter()
x = array("d", (random.random() for _ in range(N)))
print(f"Generation: {time.perf_counter() - t0:.3f} s")

times = []
for i in range(RUNS):
    t = time.perf_counter()
    std = statistics.pstdev(x)
    times.append(time.perf_counter() - t)
    print(f"Run {i}: {times[-1]:.3f} s")

times.sort()
print(f"Standard deviation: {std:.6f}")
print(f"Median: {times[RUNS // 2]:.3f} s")
