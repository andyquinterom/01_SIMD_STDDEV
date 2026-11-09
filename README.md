# Ultra-paralelismo

Code for the video: how fast can we compute the **standard deviation of
1,000,000,000 numbers**? We start with pure Python and end with Rust using
SIMD, every CPU core and cache-sized blocks.

Every Rust implementation has the same signature, lives in its own file, and
the folders follow the order of the video.

## Where is everything?

```
.
├── baselines/              Other languages, as a reference point
│   ├── python/
│   │   ├── 1_pure_python.py        hand-written loop
│   │   ├── 2_statistics_pstdev.py  standard library
│   │   └── 3_numpy.py              np.std
│   └── r/
│       └── sd.R                    sd()
│
├── src/                    Rust implementations (one file = one version)
│   ├── lib.rs              Lists every implementation, in video order
│   ├── state.rs            Welford update + Chan merge (shared by 4 and 5)
│   ├── 01_scalar/          1. Plain Rust, one thread
│   ├── 02_simd/            2. SIMD with std::simd
│   ├── 03_parallel/        3. SIMD + all cores (rayon)
│   ├── 04_welford_chan/    4. Numerically stable single pass
│   └── 05_blocks/          5. Cache-sized blocks: stable *and* fast
│
├── benches/standard_deviation.rs   Benchmarks every implementation
├── tests/all_implementations.rs    Checks every implementation is correct
└── examples/compare_block_opt.rs   A/B test: block vs. optimized block
```

## The implementations, in order

| # | File | Idea |
|---|------|------|
| 1.1 | [`src/01_scalar/sequential.rs`](src/01_scalar/sequential.rs) | Textbook two-pass: mean, then squared deviations |
| 1.2 | [`src/01_scalar/sequential_single_pass.rs`](src/01_scalar/sequential_single_pass.rs) | One pass with `Σx` and `Σx²` (fast but unstable) |
| 2.1 | [`src/02_simd/simd_single_pass.rs`](src/02_simd/simd_single_pass.rs) | Single pass, 8 SIMD lanes |
| 2.2 | [`src/02_simd/simd_double_pass_2.rs`](src/02_simd/simd_double_pass_2.rs) | Two passes, 2 SIMD lanes |
| 2.3 | [`src/02_simd/simd_double_pass_4.rs`](src/02_simd/simd_double_pass_4.rs) | Two passes, 4 SIMD lanes |
| 2.4 | [`src/02_simd/simd_double_pass_8.rs`](src/02_simd/simd_double_pass_8.rs) | Two passes, 8 SIMD lanes |
| 3.1 | [`src/03_parallel/simd_single_pass_par.rs`](src/03_parallel/simd_single_pass_par.rs) | 2.1 on every core |
| 3.2 | [`src/03_parallel/simd_double_pass_8_par.rs`](src/03_parallel/simd_double_pass_8_par.rs) | 2.4 on every core |
| 4.1 | [`src/04_welford_chan/welford.rs`](src/04_welford_chan/welford.rs) | Welford's online algorithm |
| 4.2 | [`src/04_welford_chan/welford_simd.rs`](src/04_welford_chan/welford_simd.rs) | Welford per SIMD lane, merged with Chan |
| 4.3 | [`src/04_welford_chan/modified_welford_chan_simd.rs`](src/04_welford_chan/modified_welford_chan_simd.rs) | Micro-blocks + Chan, no per-element division |
| 4.4 | [`src/04_welford_chan/modified_welford_chan_simd_par.rs`](src/04_welford_chan/modified_welford_chan_simd_par.rs) | 4.3 on every core |
| 5.1 | [`src/05_blocks/simd_shifted.rs`](src/05_blocks/simd_shifted.rs) | Single pass shifted by an estimate of the mean |
| 5.2 | [`src/05_blocks/simd_double_pass_block.rs`](src/05_blocks/simd_double_pass_block.rs) | Two passes per 32 KiB block + Chan merge |
| 5.3 | [`src/05_blocks/simd_double_pass_block_opt.rs`](src/05_blocks/simd_double_pass_block_opt.rs) | 5.2 with 4 accumulators, FMA and 16 KiB blocks |

The Chan merge formula used in chapters 4 and 5 is in
[`src/state.rs`](src/state.rs):

```
n_AB    = n_A + n_B
δ       = mean_B - mean_A
mean_AB = mean_A + δ · n_B / n_AB
M2_AB   = M2_A + M2_B + δ² · n_A · n_B / n_AB
```

## Results (N = 1,000,000,000, 12 threads)

Measured on the machine used for the video:

| Implementation | Time |
|----------------|-----:|
| R `sd()` | 1.869 s |
| `sequential` | 1.052 s |
| NumPy `np.std` | 0.953 s |
| `simd_double_pass_2` | 637 ms |
| `sequential_single_pass` | 558 ms |
| `simd_double_pass_4` | 333 ms |
| `simd_double_pass_8` | 208 ms |
| `simd_double_pass_block` | 196 ms |
| `simd_single_pass` | 110 ms |
| `simd_double_pass_8_par` | 67 ms |
| `simd_double_pass_block_par` | 37 ms |
| `simd_single_pass_par` | 34 ms |

## Running it yourself

### Requirements

- **Rust nightly** (for `std::simd`). The [`rust-toolchain.toml`](rust-toolchain.toml)
  selects it automatically if you use `rustup`.
- **[uv](https://docs.astral.sh/uv/)** for the Python baselines.
- **R** for the R baseline.
- About **8 GB of free RAM** for the default N = 1,000,000,000.

### Rust

```bash
cargo test --release
```

```bash
cargo bench
```

```bash
cargo bench -- simd_double_pass_8
```

Use less data or a different thread count with environment variables:

```bash
N=100_000_000 THREADS=8 cargo bench
```

Criterion writes HTML reports to `target/criterion/report/index.html`.

### Python and R

Every script takes an optional N (default 1,000,000,000):

```bash
cd baselines/python && uv run 3_numpy.py
```

```bash
cd baselines/python && uv run 1_pure_python.py 10_000_000
```

```bash
Rscript baselines/r/sd.R
```

## Adding your own implementation

1. Create a file in the right chapter folder with
   `pub fn standard_deviation(data: &[f64]) -> f64`.
2. Declare it in the matching `mod` block in [`src/lib.rs`](src/lib.rs).
3. Add it to `IMPLEMENTATIONS` in `src/lib.rs`.

That's it: `cargo test` checks it and `cargo bench` measures it.
