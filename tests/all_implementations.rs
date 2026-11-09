//! Correctness tests: every implementation in `IMPLEMENTATIONS` is checked
//! against hand-computed values and a stable two-pass reference.
//!
//! ```bash
//! cargo test --release
//! ```

use proptest::prelude::*;
use ultra_paralelismo::{IMPLEMENTATIONS, STABLE_IMPLEMENTATIONS};

#[test]
fn test_standard_deviation_basic_data() {
    let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    for (name, func) in IMPLEMENTATIONS {
        let result = func(&data);
        assert!(
            (result - 2.0).abs() < 1e-10,
            "Failed implementation: {}",
            name
        );
    }
}

#[test]
fn test_standard_deviation_equal_values() {
    let data = vec![5.0, 5.0, 5.0, 5.0];
    for (name, func) in IMPLEMENTATIONS {
        let result = func(&data);
        assert!(
            (result - 0.0).abs() < 1e-10,
            "Failed implementation: {}",
            name
        );
    }
}

#[test]
fn test_standard_deviation_single_element() {
    let data = vec![10.0];
    for (name, func) in IMPLEMENTATIONS {
        let result = func(&data);
        assert!(
            (result - 0.0).abs() < 1e-10,
            "Failed implementation: {}",
            name
        );
    }
}

#[test]
fn test_standard_deviation_two_elements() {
    let data = vec![3.0, 7.0];
    for (name, func) in IMPLEMENTATIONS {
        let result = func(&data);
        assert!(
            (result - 2.0).abs() < 1e-10,
            "Failed implementation: {}",
            name
        );
    }
}

#[test]
fn test_standard_deviation_negative_values() {
    let data = vec![-2.0, -4.0, -4.0, -4.0, -5.0, -5.0, -7.0, -9.0];
    for (name, func) in IMPLEMENTATIONS {
        let result = func(&data);
        assert!(
            (result - 2.0).abs() < 1e-10,
            "Failed implementation: {}",
            name
        );
    }
}

#[test]
fn test_standard_deviation_mixed_values() {
    let data = vec![-5.0, 0.0, 5.0];
    let expected = (50.0_f64 / 3.0_f64).sqrt();
    for (name, func) in IMPLEMENTATIONS {
        let result = func(&data);
        assert!(
            (result - expected).abs() < 1e-10,
            "Failed implementation: {}",
            name
        );
    }
}

/// Consistency test: all implementations must return the same result
#[test]
fn test_consistency_between_implementations() {
    let test_cases = vec![
        vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0],
        vec![5.0, 5.0, 5.0, 5.0],
        vec![10.0],
        vec![3.0, 7.0],
        vec![-2.0, -4.0, -4.0, -4.0, -5.0, -5.0, -7.0, -9.0],
        vec![-5.0, 0.0, 5.0],
    ];

    for data in test_cases {
        let results: Vec<f64> = IMPLEMENTATIONS
            .iter()
            .map(|(_, func)| func(&data))
            .collect();

        // All results must be equal
        let first = results[0];
        for (i, result) in results.iter().enumerate() {
            assert!(
                (first - result).abs() < 1e-10,
                "Inconsistency in data {:?}: impl 0 = {}, impl {} = {}",
                data,
                first,
                i,
                result
            );
        }
    }
}

// ------------------------------------------------------------------
// Helpers for deep validation
// ------------------------------------------------------------------

/// Stable two-pass reference in `f64`.
fn two_pass_reference(data: &[f64]) -> f64 {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let variance = data.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    variance.max(0.0).sqrt()
}

/// Compares with relative tolerance, avoiding demanding absolute precision
/// around zero.
fn assert_close(obtained: f64, expected: f64, tol: f64, context: &str) {
    let scale = expected.abs().max(1.0);
    assert!(
        (obtained - expected).abs() <= tol * scale,
        "{context}: obtained {obtained}, expected {expected} (tol {tol})"
    );
}

/// Deterministic (LCG) generator of values in `[0, 1)`, without dependencies.
fn lcg_values(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 11) as f64 / (1u64 << 53) as f64
        })
        .collect()
}

/// Runs all implementations against the stable reference.
fn compare_all(data: &[f64], tol: f64, context: &str) {
    let expected = two_pass_reference(data);
    for (name, func) in IMPLEMENTATIONS {
        let obtained = func(data);
        assert_close(obtained, expected, tol, &format!("{context} / {name}"));
    }
}

// ------------------------------------------------------------------
// Deep validation tests
// ------------------------------------------------------------------

/// Walks through all lengths that cross the LANES boundaries (2, 4, 8),
/// exercising the SIMD prefix and suffix.
#[test]
fn test_consistency_boundary_lengths() {
    for n in 1..=40usize {
        let data = lcg_values(n, n as u64 + 1);
        compare_all(&data, 1e-12, &format!("length {n}"));
    }
}

/// The parallel versions split into blocks of `BLOCK_SIZE`
/// (32 KiB / 8 = 4096 elements). Both sides of the boundary and a final
/// partial block are covered.
#[test]
fn test_parallel_block_boundaries() {
    for n in [4095usize, 4096, 4097, 4096 * 3 + 7] {
        let data = lcg_values(n, n as u64);
        compare_all(&data, 1e-9, &format!("block {n}"));
    }
}

/// Compares all implementations over increasing volumes of
/// pseudo-random data.
#[test]
fn test_large_random_data() {
    for n in [1_000usize, 100_000, 1_000_000] {
        let data = lcg_values(n, 0xDEAD_BEEF ^ n as u64);
        compare_all(&data, 1e-9, &format!("random {n}"));
    }
}

/// Values whose standard deviation can be computed by hand.
#[test]
fn test_known_values() {
    let cases: &[(&[f64], f64)] = &[
        (&[1.0, 2.0, 3.0, 4.0], 1.25_f64.sqrt()),
        (&[1.0, 2.0, 3.0, 4.0, 5.0], 2.0_f64.sqrt()),
        (&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], (60.0_f64 / 9.0).sqrt()),
        (&[0.0, 0.0, 0.0, 0.0], 0.0),
        (&[1.0, -1.0], 1.0),
    ];
    for (data, expected) in cases {
        for (name, func) in IMPLEMENTATIONS {
            assert_close(
                func(data),
                *expected,
                1e-12,
                &format!("known values {data:?} / {name}"),
            );
        }
    }
}

/// Catastrophic cancellation: large values that are very close to each
/// other. Double-pass implementations must remain stable; single-pass /
/// Welford ones are not compared for precision, but they are required to
/// return a finite, non-negative value (no `NaN`).
#[test]
fn test_cancellation_stability() {
    let n = 100_000usize;
    let base = 1.0e9_f64;
    let data: Vec<f64> = (0..n).map(|i| base + (i % 101) as f64).collect();

    let expected = two_pass_reference(&data);
    let stable = IMPLEMENTATIONS
        .iter()
        .filter(|(name, _)| STABLE_IMPLEMENTATIONS.contains(name));
    for (name, func) in stable {
        assert_close(
            func(&data),
            expected,
            1e-6,
            &format!("stable cancellation / {name}"),
        );
    }

    for (name, func) in IMPLEMENTATIONS {
        let obtained = func(&data);
        assert!(
            obtained.is_finite() && obtained >= 0.0,
            "cancellation / {name}: expected a finite non-negative value, obtained {obtained}"
        );
    }
}

/// Standard deviation is invariant under translation.
#[test]
fn test_translation_invariance() {
    let data = lcg_values(50_000, 7);
    let shifted: Vec<f64> = data.iter().map(|v| v + 1_000.0).collect();
    for (name, func) in IMPLEMENTATIONS {
        assert_close(
            func(&shifted),
            func(&data),
            1e-6,
            &format!("translation / {name}"),
        );
    }
}

/// Standard deviation scales linearly: `f(a·x) = |a|·f(x)`.
#[test]
fn test_scale_invariance() {
    let data = lcg_values(50_000, 11);
    let a = -3.5_f64;
    let scaled: Vec<f64> = data.iter().map(|v| v * a).collect();
    for (name, func) in IMPLEMENTATIONS {
        assert_close(
            func(&scaled),
            a.abs() * func(&data),
            1e-12,
            &format!("scale / {name}"),
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    /// Compares all implementations against the stable reference over
    /// bounded data vectors (avoiding `NaN`/`∞`).
    #[test]
    fn test_property_all_implementations(
        data in proptest::collection::vec(-1.0e6f64..1.0e6f64, 1..10_000)
    ) {
        compare_all(&data, 1e-6, "proptest");
    }
}
