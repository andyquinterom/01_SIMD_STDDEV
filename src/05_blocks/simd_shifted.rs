//! # 5.1 Shifted single pass (SIMD, optionally parallel)
//!
//! Single pass of `Σd` / `Σd²` over `d = x - k`, where `k` is an estimate of
//! the mean taken from a small sample. Shifting removes most of the
//! cancellation problem of the naive single pass.

use rayon::prelude::*;
use std::simd::{Simd, StdFloat, num::SimdFloat};

const LANES: usize = 8;
const BLOCK_SIZE: usize = 32 * 1024 / size_of::<f64>();

/// Accumulates over a block the sum of the deviations from `k` (`s1`) and the
/// sum of their squares (`s2`), using SIMD and FMA.
///
/// Subtracting a constant `k` close to the mean reduces catastrophic
/// cancellation: the shifted values `d = x - k` are small, so `d * d` keeps its
/// precision even when `x` is huge.
fn accumulate(block: &[f64], k: f64) -> (f64, f64, f64) {
    let (prefix, vectors, suffix) = block.as_simd::<LANES>();
    let kv = Simd::<f64, LANES>::splat(k);
    let mut s1 = Simd::<f64, LANES>::splat(0.0);
    let mut s2 = Simd::<f64, LANES>::splat(0.0);

    for &v in vectors {
        let d = v - kv;
        s1 += d;
        s2 = d.mul_add(d, s2); // FMA: StdFloat
    }

    let (mut a, mut b) = (s1.reduce_sum(), s2.reduce_sum());
    for &x in prefix.iter().chain(suffix) {
        let d = x - k;
        a += d;
        b += d * d;
    }
    (block.len() as f64, a, b)
}

/// Estimates `k` as the mean of a small, evenly spaced sample. The cost is
/// negligible and it is enough to bring `k` close to the real mean.
fn estimate_k(data: &[f64]) -> f64 {
    let step = (data.len() / 4096).max(1);
    let (sum, count) = data
        .iter()
        .step_by(step)
        .fold((0.0, 0.0), |(s, c), &x| (s + x, c + 1.0));
    sum / count
}

/// From `(n, s1, s2)`, computes the standard deviation correcting for the shift
/// introduced by `k`.
#[inline]
fn finalize(n: f64, s1: f64, s2: f64) -> f64 {
    let mean_d = s1 / n;
    (s2 / n - mean_d * mean_d).max(0.0).sqrt()
}

/// Single-threaded version.
pub fn standard_deviation(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let k = estimate_k(data);
    let (n, s1, s2) = accumulate(data, k);
    finalize(n, s1, s2)
}

/// Parallel version: splits the data into blocks with rayon.
pub fn standard_deviation_par(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let k = estimate_k(data);
    let (n, s1, s2) = data
        .par_chunks(BLOCK_SIZE)
        .map(|block| accumulate(block, k))
        .reduce(|| (0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));

    finalize(n, s1, s2)
}
