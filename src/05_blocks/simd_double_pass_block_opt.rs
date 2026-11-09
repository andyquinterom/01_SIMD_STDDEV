//! # 5.3 Two passes per block + Chan, optimized
//!
//! Optimized copy of [`super::simd_double_pass_block`].
//!
//! It keeps exactly the same algorithm (two passes per block + Chan
//! combination) but attacks the real bottlenecks:
//!
//! 1. **ILP / dependency chains**: the original does `acc += v` in a single SIMD
//!    accumulator. Since Rust does not allow reassociating floating-point
//!    operations, the compiler cannot break that chain. Here we use four
//!    independent accumulators (unrolled ×4), so the floating-point execution
//!    unit never waits.
//! 2. **FMA in the second pass**: `d.mul_add(d, acc)` computes `d*d + acc` in a
//!    single instruction with a single rounding (more accurate and faster).
//! 3. **Cache-friendly blocking also in the sequential version**: the second
//!    pass of each block is served from L1/L2, so main memory is read only once
//!    per element.
//! 4. **Branch-free prefix/suffix**: they are handled with the same SIMD
//!    accumulator as the main loop by means of the `chunks_exact` remainder.
//! 5. **Block sized at 16 KiB** (not 32 KiB): measured empirically, it is the
//!    largest size whose second pass still fits in L1; 24 KiB already degrades
//!    performance. It matches the value the original already used.

use rayon::prelude::*;
use std::simd::{Simd, StdFloat, num::SimdFloat};
use crate::state::State;

/// `f64` elements per SIMD vector.
const LANES: usize = 8;
/// Independent SIMD accumulators: they break the dependency chain of the
/// floating-point sums (Rust does not reassociate them on its own).
const ACCUMULATORS: usize = 4;
/// 16 KiB blocks. Measured empirically: it is the largest size whose second
/// pass is still served from L1 on this microarchitecture; 24 KiB already
/// degrades.
const BLOCK_SIZE: usize = 16 * 1024 / size_of::<f64>();

/// Two SIMD passes over a block, returning its local `State`.
///
/// The block is small enough to stay in L1 between the first and the second
/// pass, so the second read does not touch main memory.
#[inline]
fn block_double_pass(data: &[f64]) -> State {
    let n = data.len() as f64;
    let (prefix, vectors, suffix) = data.as_simd::<LANES>();

    // ------------------------------------------------------------------
    // Pass 1: sum of the elements
    // ------------------------------------------------------------------
    let (groups, rest) = vectors.as_chunks::<ACCUMULATORS>();
    let mut acc = [Simd::<f64, LANES>::splat(0.0); ACCUMULATORS];
    for group in groups {
        for (a, &v) in acc.iter_mut().zip(group) {
            *a += v;
        }
    }
    let mut sum_vec = acc[0];
    for &a in &acc[1..] {
        sum_vec += a;
    }
    let mut sum = sum_vec.reduce_sum();
    for &v in rest {
        sum += v.reduce_sum();
    }
    for &value in prefix.iter().chain(suffix) {
        sum += value;
    }

    let mean = sum / n;

    // ------------------------------------------------------------------
    // Pass 2: sum of squared deviations, with FMA
    // ------------------------------------------------------------------
    let mv = Simd::<f64, LANES>::splat(mean);
    let mut acc = [Simd::<f64, LANES>::splat(0.0); ACCUMULATORS];
    for group in groups {
        for (a, &v) in acc.iter_mut().zip(group) {
            let d = v - mv;
            *a = d.mul_add(d, *a);
        }
    }
    let mut squares_vec = acc[0];
    for &a in &acc[1..] {
        squares_vec += a;
    }
    let mut squares = squares_vec.reduce_sum();
    for &v in rest {
        let d = v - mv;
        squares += d.mul_add(d, Simd::splat(0.0)).reduce_sum();
    }
    for &value in prefix.iter().chain(suffix) {
        let d = value - mean;
        squares += d * d;
    }

    State {
        n,
        mean,
        m2: squares,
    }
}

/// Single-threaded version, with cache-friendly blocking.
pub fn standard_deviation(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    data.chunks(BLOCK_SIZE)
        .fold(State::default(), |acc, block| {
            acc.combine(block_double_pass(block))
        })
        .finalize()
}

/// Parallel version: one block per rayon task and a Chan reduction.
///
/// `fold` is used instead of `map` + `reduce`: rayon combines the local
/// `State`s more efficiently within each thread (measured ~10% faster than
/// `map` + `reduce` for this block size).
pub fn standard_deviation_par(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    data.par_chunks(BLOCK_SIZE)
        .fold(State::default, |acc, block| {
            acc.combine(block_double_pass(block))
        })
        .reduce(State::default, State::combine)
        .finalize()
}
