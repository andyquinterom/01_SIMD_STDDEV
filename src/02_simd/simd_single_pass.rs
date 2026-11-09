//! # 2.1 SIMD, single pass (8 lanes)
//!
//! Same `Σx` / `Σx²` idea as `sequential_single_pass`, but processing 8 `f64`
//! at a time with `std::simd`. Inherits the same numerical instability.

use std::simd::{Simd, num::SimdFloat};

const LANES: usize = 8;

pub fn standard_deviation(data: &[f64]) -> f64 {
    let n = data.len() as f64;

    let (prefix, vectors, suffix) = data.as_simd::<LANES>();

    let mut sum_vec = Simd::<f64, LANES>::splat(0.0);
    let mut sum_squares_vec = Simd::<f64, LANES>::splat(0.0);

    for &vector in vectors {
        sum_vec += vector;
        sum_squares_vec += vector * vector;
    }

    let mut sum = sum_vec.reduce_sum();
    let mut sum_of_squares = sum_squares_vec.reduce_sum();

    for &value in prefix.iter().chain(suffix) {
        sum += value;
        sum_of_squares += value * value;
    }

    let mean = sum / n;
    let variance = (sum_of_squares / n) - mean * mean;
    variance.max(0.0).sqrt()
}
