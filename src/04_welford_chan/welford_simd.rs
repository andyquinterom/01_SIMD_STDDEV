//! # 4.2 Welford, SIMD
//!
//! Each of the 8 SIMD lanes runs its own Welford recurrence; at the end the
//! lanes are merged with Chan's formula (`State::combine`).

use std::simd::Simd;
use crate::state::State;

const LANES: usize = 8;

pub fn standard_deviation(data: &[f64]) -> f64 {
    let (prefix, vectors, suffix) = data.as_simd::<LANES>();

    // Each lane keeps its own Welford state, using the same recurrence as the
    // scalar version but vectorized: element `i` of the vector updates lane `i`.
    // Thus each lane accumulates the subsequence `i, i + LANES, i + 2 * LANES, ...`.
    let mut n = Simd::splat(0.0);
    let mut mean = Simd::<f64, LANES>::splat(0.0);
    let mut m2 = Simd::<f64, LANES>::splat(0.0);

    for &vector in vectors {
        n += Simd::splat(1.0);
        let delta = vector - mean;
        mean += delta / n;
        let delta2 = vector - mean;
        m2 += delta * delta2;
    }

    // Lanes are combined with Chan's formula, just like in the parallel version.
    let means = mean.to_array();
    let m2_values = m2.to_array();
    let n = n.to_array()[0];

    let mut state = State::default();
    for i in 0..LANES {
        state = state.combine(State {
            n,
            mean: means[i],
            m2: m2_values[i],
        });
    }

    // Elements that do not fill a whole vector are processed with the scalar
    // recurrence.
    for &value in prefix.iter().chain(suffix) {
        state = state.welford(value);
    }

    state.finalize()
}
