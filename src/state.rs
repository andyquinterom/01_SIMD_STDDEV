//! Running statistics shared by the Welford and block-based implementations.
//!
//! A `State` summarizes a set of values with three numbers:
//!
//! - `n`:    how many values we have seen
//! - `mean`: their mean
//! - `m2`:   the sum of squared deviations from that mean, `Σ (x - mean)²`
//!
//! The variance is `m2 / n` and the standard deviation is its square root.

#[derive(Clone, Copy, Default, Debug)]
pub struct State {
    pub n: f64,
    pub mean: f64,
    pub m2: f64,
}

impl State {
    /// Welford's update: adds a single value to the running state.
    #[inline]
    pub fn welford(self, value: f64) -> Self {
        let n = self.n + 1.0;
        let delta = value - self.mean;
        let mean = self.mean + delta / n;
        let delta2 = value - mean;
        Self {
            n,
            mean,
            m2: self.m2 + delta * delta2,
        }
    }

    /// Chan's formula: merges two partial states (A and B) into one.
    ///
    /// ```text
    /// n_AB  = n_A + n_B
    /// δ     = mean_B - mean_A
    /// mean_AB = mean_A + δ · n_B / n_AB
    /// M2_AB = M2_A + M2_B + δ² · n_A · n_B / n_AB
    /// ```
    ///
    /// This is what lets us split the data into blocks, process them
    /// independently (in SIMD lanes or on different threads) and combine the
    /// results at the end.
    #[inline]
    pub fn combine(self, other: Self) -> Self {
        if self.n == 0.0 {
            return other;
        }
        if other.n == 0.0 {
            return self;
        }
        let n = self.n + other.n;
        let delta = other.mean - self.mean;
        let mean = self.mean + delta * (other.n / n);
        let m2 = self.m2 + other.m2 + delta * delta * (self.n * other.n / n);
        Self { n, mean, m2 }
    }

    /// Population standard deviation: `sqrt(m2 / n)`.
    #[inline]
    pub fn finalize(self) -> f64 {
        if self.n == 0.0 {
            return 0.0;
        }
        (self.m2 / self.n).max(0.0).sqrt()
    }
}
