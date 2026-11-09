//! # 1.1 Sequential, two passes
//!
//! The textbook definition: one pass to compute the mean, a second pass to sum
//! the squared deviations from it. Scalar code, a single thread.

use std::ops::Div;

pub fn standard_deviation(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let sum = data.iter().sum::<f64>();
    let mean = sum / n;

    data.iter()
        .map(|value| (value - mean).powf(2.0))
        .sum::<f64>()
        .div(n)
        .sqrt()
}
