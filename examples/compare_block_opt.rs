//! Lightweight A/B comparison between `simd_double_pass_block` (original) and
//! its optimized copy `simd_double_pass_block_opt`.
//!
//! Usage: `cargo run --release --example compare_block_opt [num_elements]`
//! (200_000_000 ≈ 1.6 GB by default).

use std::hint::black_box;
use std::time::Instant;
use ultra_paralelismo::{simd_double_pass_block as orig, simd_double_pass_block_opt as opt};

fn time<F: Fn() -> f64>(f: F) -> f64 {
    let t = Instant::now();
    let r = black_box(f());
    let dt = t.elapsed().as_secs_f64();
    black_box(r);
    dt
}

fn best<F: Fn() -> f64>(reps: usize, f: F) -> f64 {
    let mut t = f64::MAX;
    for _ in 0..reps {
        t = t.min(time(&f));
    }
    t
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000_000);

    rayon::ThreadPoolBuilder::new().num_threads(12).build_global().unwrap();

    let data: Vec<f64> = (0..n).map(|_| rand::random::<f64>()).collect();
    let gb = (n * 8) as f64 / 1e9;

    let reps = 7;

    // calibration: a single pass of pure summation
    let sum = || {
        let mut s = 0.0f64;
        for &x in &data {
            s += black_box(x);
        }
        s
    };
    let sum_par = || data.par_iter().map(|&x| x).sum::<f64>();
    use rayon::prelude::*;

    let ts = best(reps, sum);
    let tsp = best(reps, sum_par);
    println!("n={n} ({gb:.2} GB)");
    println!("  seq sum 1 pass   : {ts:>8.4} s  ({:.1} GB/s)", gb / ts);
    println!("  par sum 1 pass   : {tsp:>8.4} s  ({:.1} GB/s)", gb / tsp);

    let o_s = best(reps, || orig::standard_deviation(&data));
    let p_s = best(reps, || opt::standard_deviation(&data));
    let o_p = best(reps, || orig::standard_deviation_par(&data));
    let p_p = best(reps, || opt::standard_deviation_par(&data));

    println!("  orig seq : {o_s:>8.4} s  ({:.1} effective GB/s)", 2.0 * gb / o_s);
    println!("  opt  seq : {p_s:>8.4} s  ({:.1} effective GB/s)", 2.0 * gb / p_s);
    println!("  orig par : {o_p:>8.4} s  ({:.1} effective GB/s)", 2.0 * gb / o_p);
    println!("  opt  par : {p_p:>8.4} s  ({:.1} effective GB/s)", 2.0 * gb / p_p);

    let a = orig::standard_deviation(&data);
    let b = opt::standard_deviation(&data);
    let c = orig::standard_deviation_par(&data);
    let d = opt::standard_deviation_par(&data);
    println!("  values: orig_seq={a:.12} opt_seq={b:.12} orig_par={c:.12} opt_par={d:.12}");
}
