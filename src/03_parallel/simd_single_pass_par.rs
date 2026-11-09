//! # 3.1 SIMD + threads, single pass
//!
//! `simd_single_pass` run over 32 KiB blocks in parallel with rayon. Each block
//! returns `(Σx, Σx²)` and the partial sums are added together.

use rayon::prelude::*;
use std::simd::{Simd, num::SimdFloat};

const LANES: usize = 8;
const BLOCK_SIZE: usize = 32 * 1024 / size_of::<f64>();


pub fn standard_deviation(data: &[f64]) -> f64 {
    let n = data.len() as f64;

    let (sum, sum_of_squares) = data
        .par_chunks(BLOCK_SIZE)
        .map(|block| {
            let (prefix, vectors, suffix) = block.as_simd::<LANES>();

            let mut sum_vec = Simd::splat(0.0);
            let mut sum_squares_vec = Simd::splat(0.0);

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

            (sum, sum_of_squares)
        })
        .reduce(
            || (0.0, 0.0),
            |(sum_a, sum_squares_a), (sum_b, sum_squares_b)| {
                (sum_a + sum_b, sum_squares_a + sum_squares_b)
            },
        );

    let mean = sum / n;
    let variance = (sum_of_squares / n) - mean * mean;
    variance.max(0.0).sqrt()
}
