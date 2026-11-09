//! # 2.2 SIMD, two passes, 2 lanes
//!
//! The stable two-pass algorithm, vectorized with 2-wide `f64` vectors.
//! `as_simd` splits the slice into an unaligned `prefix`, aligned `vectors`
//! and a leftover `suffix`.

use std::simd::{Simd, num::SimdFloat};

const LANES: usize = 2;

pub fn standard_deviation(data: &[f64]) -> f64 {
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

    (sum_of_squares / n).sqrt()
}
