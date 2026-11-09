//! # 4.3 Modified Welford + Chan, SIMD
//!
//! Avoids the per-element division of Welford: data is processed in
//! micro-blocks of vectors, each one shifted by a reference value and summed
//! with plain (FMA) accumulators, then merged into the running state with
//! Chan's formula.

use std::simd::{Simd, StdFloat};
use crate::state::State;

const LANES: usize = 8;
const VECTORS_PER_MICRO_BLOCK: usize = 128;

fn welford_block(block: &[f64]) -> State {
    let (prefix, vectors, suffix) = block.as_simd::<LANES>();

    let mut n_run = 0.0;
    let mut mean_run = Simd::<f64, LANES>::splat(0.0);
    let mut m2_run = Simd::<f64, LANES>::splat(0.0);

    for group in vectors.chunks(VECTORS_PER_MICRO_BLOCK) {
        // Shift by the first vector of the group: each lane uses its own real
        // datum as the reference, which avoids catastrophic cancellation when
        // accumulating the squares.
        let reference = group[0];
        let mut sum_vec = Simd::<f64, LANES>::splat(0.0);
        let mut sum_squares_vec = Simd::<f64, LANES>::splat(0.0);

        for &vector in group {
            let d = vector - reference;
            sum_vec += d;
            sum_squares_vec = d.mul_add(d, sum_squares_vec);
        }

        let k = group.len() as f64;
        let mean_d = sum_vec * Simd::splat(1.0 / k);
        let m2_block = sum_squares_vec - sum_vec * mean_d;
        let mean_block = reference + mean_d;

        if n_run == 0.0 {
            mean_run = mean_block;
            m2_run = m2_block;
            n_run = k;
        } else {
            let new_n = n_run + k;
            let delta = mean_block - mean_run;
            mean_run += delta * Simd::splat(k / new_n);
            m2_run += m2_block + delta * delta * Simd::splat(n_run * k / new_n);
            n_run = new_n;
        }
    }

    let means = mean_run.to_array();
    let m2_values = m2_run.to_array();

    let mut state = State::default();
    for i in 0..LANES {
        state = state.combine(State {
            n: n_run,
            mean: means[i],
            m2: m2_values[i],
        });
    }

    for &value in prefix.iter().chain(suffix) {
        state = state.welford(value);
    }

    state
}

pub fn standard_deviation(data: &[f64]) -> f64 {
    let state = welford_block(data);

    state.finalize()
}
