//! Seeded property-style coverage for shape arithmetic (first slice of #48).
//!
//! No new dependencies: a tiny xorshift generator drives hundreds of
//! broadcast/reshape/matmul cases against a naive model of the NumPy rules.
//! Every case asserts the same contract — success with exactly the naive
//! geometry and values, or a typed refusal — so a shape-algebra regression
//! fails here before it can surface as a wrong kernel launch or a silent
//! miscompile.
#![cfg(feature = "cpu")]

use incin_core::prelude::*;

type B = incin_backends::cpu::CpuBackendImpl;

/// Deterministic xorshift64* with a fixed seed per test. No RNG crate.
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }

    /// 1..=4: broadcast-heavy, zero-size stays in `zero_sized_semantics.rs`.
    fn dim(&mut self) -> usize {
        1 + self.below(4)
    }

    fn shape(&mut self, max_rank: usize) -> Vec<usize> {
        (0..self.below(max_rank + 1)).map(|_| self.dim()).collect()
    }
}

/// Right-aligned 1-or-equal rule. `None` means incompatible.
fn naive_broadcast(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let x = if i < a.len() { a[a.len() - 1 - i] } else { 1 };
        let y = if i < b.len() { b[b.len() - 1 - i] } else { 1 };
        if x == y {
            out.push(x);
        } else if x == 1 {
            out.push(y);
        } else if y == 1 {
            out.push(x);
        } else {
            return None;
        }
    }
    out.reverse();
    Some(out)
}

fn dyn_ones(dims: &[usize]) -> Tensor<Dyn, B> {
    Tensor::<Dyn, B>::ones(dims.to_vec()).unwrap()
}

#[test]
fn broadcast_matches_naive_rule_on_seeded_shapes() {
    let mut rng = Lcg(0x9E3779B97F4A7C15);
    for case in 0..300 {
        let mut a = rng.shape(3);
        let mut b = rng.shape(3);
        // Every fifth case forces a shared incompatible axis so refusals get
        // as much coverage as successes.
        if case % 5 == 4 && !a.is_empty() && !b.is_empty() {
            let last_a = a.len() - 1;
            let last_b = b.len() - 1;
            a[last_a] = 2;
            b[last_b] = 3;
        }
        let got = dyn_ones(&a).broadcast_add(&dyn_ones(&b));
        match (naive_broadcast(&a, &b), got) {
            (Some(expected), Ok(tensor)) => {
                assert_eq!(
                    tensor.dims().dims(),
                    expected.as_slice(),
                    "case {case}: {a:?} vs {b:?}"
                );
                // ones + ones is twos everywhere: geometry and values agree.
                for v in tensor.to_vec1::<f32>().unwrap() {
                    assert_eq!(v, 2.0, "case {case}: {a:?} vs {b:?}");
                }
            }
            (None, Err(_)) => {}
            (Some(expected), Err(error)) => {
                panic!("case {case}: {a:?} vs {b:?} should give {expected:?}, got {error:?}")
            }
            (None, Ok(tensor)) => panic!(
                "case {case}: {a:?} vs {b:?} are incompatible but produced {:?}",
                tensor.dims().dims()
            ),
        }
    }
}

/// Split `n` into a random rank-1..=3 factorisation (1s pad the rank out).
fn random_factorisation(rng: &mut Lcg, n: usize) -> Vec<usize> {
    let mut factors = vec![n];
    while factors.len() < 3 && rng.below(2) == 0 {
        let i = rng.below(factors.len());
        let f = factors[i];
        if f < 2 {
            break;
        }
        // Random divisor pair of f (1 is always available, so this ends).
        let mut d = 1 + rng.below(f);
        while f % d != 0 {
            d -= 1;
        }
        factors[i] = d;
        factors.push(f / d);
    }
    while factors.len() < 1 + rng.below(3) {
        factors.push(1);
    }
    factors
}

#[test]
fn reshape_accepts_equal_numel_and_refuses_the_rest() {
    let mut rng = Lcg(0xD1B54A32D192ED03);
    for case in 0..200 {
        let base = {
            let mut shape = rng.shape(3);
            if shape.is_empty() {
                shape.push(2);
            }
            // Keep dim values small so the element count stays tiny.
            shape.iter().map(|_| 1 + rng.below(3)).collect::<Vec<_>>()
        };
        let numel: usize = base.iter().product();
        let target = if case % 2 == 0 {
            random_factorisation(&mut rng, numel)
        } else {
            let mut shape = rng.shape(3);
            if shape.iter().product::<usize>() == numel {
                shape.push(2);
            }
            shape
        };
        let target_numel: usize = target.iter().product();
        match dyn_ones(&base).reshape(target.clone()) {
            Ok(tensor) => {
                assert_eq!(
                    target_numel, numel,
                    "case {case}: {base:?} reshaped to {target:?} with different element counts"
                );
                assert_eq!(tensor.dims().dims(), target.as_slice());
                for v in tensor.to_vec1::<f32>().unwrap() {
                    assert_eq!(v, 1.0);
                }
            }
            Err(_) => {
                assert_ne!(
                    target_numel, numel,
                    "case {case}: {base:?} refused a same-count reshape to {target:?}"
                );
            }
        }
    }
}

#[test]
fn matmul_2d_contract_on_seeded_shapes() {
    let mut rng = Lcg(0xABC98388FB8FAC03);
    for case in 0..200 {
        let (m, k, n) = (1 + rng.below(4), 1 + rng.below(4), 1 + rng.below(4));
        // Half the cases corrupt the contracted axis (staying >= 1).
        let k2 = if case % 2 == 0 {
            k
        } else if k > 1 {
            k - 1
        } else {
            k + 1
        };
        let got = dyn_ones(&[m, k]).matmul(&dyn_ones(&[k2, n]));
        match got {
            Ok(tensor) => {
                assert_eq!(k2, k, "case {case}: mismatched inner dims executed");
                assert_eq!(tensor.dims().dims(), &[m, n]);
                // ones(m,k) @ ones(k,n) is k everywhere.
                for v in tensor.to_vec1::<f32>().unwrap() {
                    assert_eq!(v, k as f32, "case {case}");
                }
            }
            Err(_) => {
                assert_ne!(k2, k, "case {case}: matched inner dims refused");
            }
        }
    }
}

#[test]
fn transpose_swaps_axes_on_seeded_shapes() {
    let mut rng = Lcg(0x51ED5409769A9B35);
    for case in 0..200 {
        // Rank 2..=3, dims 1..=4.
        let rank = 2 + rng.below(2);
        let dims: Vec<usize> = (0..rank).map(|_| 1 + rng.below(4)).collect();
        let (a, b) = (rng.below(rank), rng.below(rank));
        if a == b {
            continue;
        }
        let mut expected = dims.clone();
        expected.swap(a, b);
        let t = dyn_ones(&dims);
        let got = t.transpose(a as isize, b as isize).unwrap_or_else(|error| {
            panic!("case {case}: transpose {dims:?} over {a},{b} refused: {error:?}")
        });
        assert_eq!(got.dims().dims(), expected.as_slice());
        for v in got.to_vec1::<f32>().unwrap() {
            assert_eq!(v, 1.0);
        }
    }
}

#[test]
fn concat_sums_the_axis_on_seeded_shapes() {
    let mut rng = Lcg(0x1F3A5C6D7E8F9A0B);
    for case in 0..200 {
        // Rank 1..=3; both operands share every axis except the concat one.
        let rank = 1 + rng.below(3);
        let axis = rng.below(rank);
        let a: Vec<usize> = (0..rank).map(|_| 1 + rng.below(4)).collect();
        let mut b = a.clone();
        b[axis] = 1 + rng.below(4);
        let mut expected = a.clone();
        expected[axis] += b[axis];
        let got = dyn_ones(&a)
            .concat(&dyn_ones(&b), axis as isize)
            .unwrap_or_else(|error| {
                panic!("case {case}: concat {a:?} with {b:?} over {axis} refused: {error:?}")
            });
        assert_eq!(got.dims().dims(), expected.as_slice());
        for v in got.to_vec1::<f32>().unwrap() {
            assert_eq!(v, 1.0);
        }
    }
}
