//! # 5.2 Two passes per block + Chan (SIMD, optionally parallel)
//!
//! Runs the stable two-pass algorithm on each 32 KiB block (which stays in
//! cache between both passes) and merges the per-block results with Chan's
//! formula. Stable *and* reads main memory only once.

use rayon::prelude::*;
use std::simd::{Simd, num::SimdFloat};
use crate::state::State;

const LANES: usize = 8;
const BLOCK_SIZE: usize = 32 * 1024 / size_of::<f64>();

fn block_double_pass(data: &[f64]) -> State {
    let n = data.len() as f64;

    let (prefix, vectors, suffix) = data.as_simd::<LANES>();

    let mut sum_vec = Simd::splat(0.0);
    for &vector in vectors {
        sum_vec += vector;
    }

    let mut sum = sum_vec.reduce_sum();
    for &value in prefix.iter().chain(suffix) {
        sum += value;
    }

    let mean = sum / n;

    let mean_vec = Simd::<f64, LANES>::splat(mean);
    let mut sum_squares_vec = Simd::splat(0.0);
    for &vector in vectors {
        let diff = vector - mean_vec;
        sum_squares_vec += diff * diff;
    }

    let mut sum_of_squares = sum_squares_vec.reduce_sum();
    for &value in prefix.iter().chain(suffix) {
        let diff = value - mean;
        sum_of_squares += diff * diff;
    }

    State {
        n,
        mean,
        m2: sum_of_squares,
    }
}

pub fn standard_deviation(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    block_double_pass(data).finalize()
}

pub fn standard_deviation_par(data: &[f64]) -> f64 {
    data
        .par_chunks(BLOCK_SIZE)
        .map(block_double_pass)
        .reduce(State::default, State::combine)
        .finalize()
}
