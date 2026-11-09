//! # Ultra-paralelismo: computing a standard deviation, fast
//!
//! Every implementation lives in its own file and exposes the same function:
//!
//! ```ignore
//! pub fn standard_deviation(data: &[f64]) -> f64
//! ```
//!
//! (some also expose `standard_deviation_par`, the multithreaded version).
//!
//! The folders follow the order of the video:
//!
//! | Folder             | Idea                                               |
//! |--------------------|----------------------------------------------------|
//! | `01_scalar/`       | Plain Rust, one thread, no SIMD                    |
//! | `02_simd/`         | Vectorized with `std::simd` (2, 4 and 8 lanes)     |
//! | `03_parallel/`     | SIMD + all cores with `rayon`                      |
//! | `04_welford_chan/` | Numerically stable single pass (Welford + Chan)    |
//! | `05_blocks/`       | Cache-sized blocks: stable *and* fast              |
//!
//! `state.rs` holds the `State` struct (Welford update + Chan merge) shared by
//! chapters 4 and 5.

#![feature(portable_simd)]

pub mod state;

#[path = "01_scalar"]
pub mod scalar {
    pub mod sequential;
    pub mod sequential_single_pass;
}

#[path = "02_simd"]
pub mod simd {
    pub mod simd_single_pass;
    pub mod simd_double_pass_2;
    pub mod simd_double_pass_4;
    pub mod simd_double_pass_8;
}

#[path = "03_parallel"]
pub mod parallel {
    pub mod simd_single_pass_par;
    pub mod simd_double_pass_8_par;
}

#[path = "04_welford_chan"]
pub mod welford_chan {
    pub mod welford;
    pub mod welford_simd;
    pub mod modified_welford_chan_simd;
    pub mod modified_welford_chan_simd_par;
}

#[path = "05_blocks"]
pub mod blocks {
    pub mod simd_shifted;
    pub mod simd_double_pass_block;
    pub mod simd_double_pass_block_opt;
}

// Flat re-exports, so every implementation can be reached as
// `ultra_paralelismo::<name>` regardless of its chapter.
pub use blocks::{simd_double_pass_block, simd_double_pass_block_opt, simd_shifted};
pub use parallel::{simd_double_pass_8_par, simd_single_pass_par};
pub use scalar::{sequential, sequential_single_pass};
pub use simd::{simd_double_pass_2, simd_double_pass_4, simd_double_pass_8, simd_single_pass};
pub use welford_chan::{
    modified_welford_chan_simd, modified_welford_chan_simd_par, welford, welford_simd,
};

/// A named implementation: `(name, function)`.
pub type Implementation = (&'static str, fn(&[f64]) -> f64);

/// Every implementation, in the order they appear in the video.
///
/// The benchmark and the tests both iterate over this list, so adding a new
/// implementation here is all it takes to benchmark and test it.
pub const IMPLEMENTATIONS: &[Implementation] = &[
    // 1. Scalar
    ("sequential", sequential::standard_deviation),
    ("sequential_single_pass", sequential_single_pass::standard_deviation),
    // 2. SIMD
    ("simd_single_pass", simd_single_pass::standard_deviation),
    ("simd_double_pass_2", simd_double_pass_2::standard_deviation),
    ("simd_double_pass_4", simd_double_pass_4::standard_deviation),
    ("simd_double_pass_8", simd_double_pass_8::standard_deviation),
    // 3. SIMD + threads
    ("simd_single_pass_par", simd_single_pass_par::standard_deviation),
    ("simd_double_pass_8_par", simd_double_pass_8_par::standard_deviation),
    // 4. Welford + Chan
    ("welford", welford::standard_deviation),
    ("welford_simd", welford_simd::standard_deviation),
    ("modified_welford_chan_simd", modified_welford_chan_simd::standard_deviation),
    ("modified_welford_chan_simd_par", modified_welford_chan_simd_par::standard_deviation),
    // 5. Cache-sized blocks
    ("simd_shifted", simd_shifted::standard_deviation),
    ("simd_shifted_par", simd_shifted::standard_deviation_par),
    ("simd_double_pass_block", simd_double_pass_block::standard_deviation),
    ("simd_double_pass_block_par", simd_double_pass_block::standard_deviation_par),
    ("simd_double_pass_block_opt", simd_double_pass_block_opt::standard_deviation),
    ("simd_double_pass_block_opt_par", simd_double_pass_block_opt::standard_deviation_par),
];

/// Implementations that are numerically stable (two passes, shifted, or
/// Chan-merged blocks). The naive single-pass ones are excluded.
pub const STABLE_IMPLEMENTATIONS: &[&str] = &[
    "sequential",
    "simd_double_pass_2",
    "simd_double_pass_4",
    "simd_double_pass_8",
    "simd_double_pass_8_par",
    "simd_shifted",
    "simd_shifted_par",
    "simd_double_pass_block",
    "simd_double_pass_block_par",
    "simd_double_pass_block_opt",
    "simd_double_pass_block_opt_par",
];
