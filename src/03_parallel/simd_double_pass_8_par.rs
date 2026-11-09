//! # 3.2 SIMD + threads, two passes
//!
//! `simd_double_pass_8` run in parallel with rayon: a parallel pass for the
//! mean, then a parallel pass for the squared deviations. Each pass reads all
//! the data from main memory, so this is limited by memory bandwidth.

use rayon::prelude::*;
use std::simd::{Simd, num::SimdFloat};

const LANES: usize = 8;
const BLOCK_SIZE: usize = 32 * 1024 / size_of::<f64>();

pub fn standard_deviation(data: &[f64]) -> f64 {
    let n = data.len() as f64;

    let sum = data
        .par_chunks(BLOCK_SIZE)
        .map(|block| {
            let (prefix, vectors, suffix) = block.as_simd::<LANES>();

            let mut sum_vec = Simd::splat(0.0);
            for &vector in vectors {
                sum_vec += vector;
            }

            let mut sum = sum_vec.reduce_sum();
            for &value in prefix.iter().chain(suffix) {
                sum += value;
            }

            sum
        })
        .reduce(|| 0.0, |a, b| a + b);

    let mean = sum / n;

    let mean_vec = Simd::<f64, LANES>::splat(mean);
    let sum_of_squares = data
        .par_chunks(BLOCK_SIZE)
        .map(|block| {
            let (prefix, vectors, suffix) = block.as_simd::<LANES>();

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

            sum_of_squares
        })
        .reduce(|| 0.0, |a, b| a + b);

    (sum_of_squares / n).sqrt()
}
