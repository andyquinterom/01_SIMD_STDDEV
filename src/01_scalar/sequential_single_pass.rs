//! # 1.2 Sequential, single pass
//!
//! Reads the data only once by accumulating `Σx` and `Σx²` at the same time,
//! then uses `Var = E[x²] - E[x]²`. Faster, but numerically unstable: when the
//! values are large and close together the subtraction cancels catastrophically.

pub fn standard_deviation(data: &[f64]) -> f64 {
    let n = data.len() as f64;

    let (sum, sum_of_squares) = data
        .iter()
        .fold((0.0, 0.0), |(sum, sum_of_squares), &value| {
            (sum + value, sum_of_squares + value * value)
        });

    let mean = sum / n;
    let variance = (sum_of_squares / n) - mean * mean;
    variance.max(0.0).sqrt()
}
