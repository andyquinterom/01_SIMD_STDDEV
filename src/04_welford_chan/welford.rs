//! # 4.1 Welford (scalar)
//!
//! Welford's online algorithm: a single, numerically stable pass that updates
//! the running mean and `M2` one value at a time. See [`crate::state::State`].

use crate::state::State;

pub fn standard_deviation(data: &[f64]) -> f64 {
    let mut state = State::default();
    for &value in data {
        state = state.welford(value);
    }

    state.finalize()
}
