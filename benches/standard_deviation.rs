//! Benchmarks every implementation in `IMPLEMENTATIONS` on the same data.
//!
//! ```bash
//! cargo bench                                  # all implementations
//! cargo bench -- simd_double_pass_8            # names containing this text
//! N=100000000 THREADS=8 cargo bench            # smaller data / fewer threads
//! ```
//!
//! By default: N = 1,000,000,000 random `f64` values (8 GB of RAM) and 12 threads.

use criterion::{Criterion, criterion_group, criterion_main};
use std::{hint::black_box, time::Duration};
use ultra_paralelismo::IMPLEMENTATIONS;

fn env_or(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.replace('_', "").parse().ok())
        .unwrap_or(default)
}

fn configure_criterion() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .measurement_time(Duration::from_secs(10))
}

fn bench_all(c: &mut Criterion) {
    let n = env_or("N", 1_000_000_000);
    let threads = env_or("THREADS", 12);

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap();

    println!("Generating {n} random f64 values ({threads} threads)...");
    let data: Vec<f64> = (0..n).map(|_| rand::random::<f64>()).collect();

    for &(name, func) in IMPLEMENTATIONS {
        c.bench_function(name, |b| b.iter(|| func(black_box(&data))));
    }
}

criterion_group!(
    name = benches;
    config = configure_criterion();
    targets = bench_all
);
criterion_main!(benches);
