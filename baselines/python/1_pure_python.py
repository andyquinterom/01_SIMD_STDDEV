"""Baseline 1: pure Python, single thread, hand-written two-pass algorithm.

Usage: uv run 1_pure_python.py [N]      (default N = 1,000,000,000)
"""

import math
import random
import sys
import time
from array import array

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000_000
RUNS = 10


def standard_deviation(data):
    n = len(data)
    mean = sum(data) / n
    m2 = 0.0
    for x in data:
        diff = x - mean
        m2 += diff * diff
    return math.sqrt(m2 / n)


# array('d') stores contiguous doubles (8 bytes per element), like a Vec<f64>.
print(f"Generating {N:,} random observations (float64)...")
t0 = time.perf_counter()
x = array("d", (random.random() for _ in range(N)))
print(f"Generation: {time.perf_counter() - t0:.3f} s")

times = []
for i in range(RUNS):
    t = time.perf_counter()
    std = standard_deviation(x)
    times.append(time.perf_counter() - t)
    print(f"Run {i}: {times[-1]:.3f} s")

times.sort()
print(f"Standard deviation: {std:.6f}")
print(f"Median: {times[RUNS // 2]:.3f} s")
