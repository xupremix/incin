//! `SHP-007`: the broadcast rule against a reference, over generated shapes.
//!
//! The typed families are checked by naming shapes one at a time, which is
//! exact but only covers the pairs someone thought to write down. This file
//! covers the rest by generating them, and runs everything through `Dyn`, the
//! one shape whose rank and axes are both free at runtime.
//!
//! Randomized cases use a fixed-seed xorshift generator rather than a
//! property-testing dependency, so a failure reproduces from the seed printed
//! in the assertion and the suite adds nothing to the dependency graph. The
//! dimension distribution is deliberately not uniform: 0 and 1 are the values
//! the broadcast rule actually turns on, so they are drawn far more often than
//! chance would give.

use incin_core::prelude::{BroadcastShape, Dyn, DynShape, Shape};

// --- deterministic generator -------------------------------------------

struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        // Any nonzero state works; xorshift64 is degenerate at 0.
        Self(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }

    /// A dimension drawn from the values the broadcast rule branches on, not
    /// from a uniform range where 0 and 1 would essentially never appear.
    fn dim(&mut self) -> usize {
        match self.below(8) {
            0 | 1 | 2 => 1,
            3 => 0,
            _ => self.below(6) as usize + 2,
        }
    }

    fn shape(&mut self, max_rank: usize) -> Vec<usize> {
        let rank = self.below(max_rank as u64 + 1) as usize;
        (0..rank).map(|_| self.dim()).collect()
    }
}

// --- reference implementation -------------------------------------------

/// NumPy's rule, written out plainly: right-align, treat a missing axis as 1,
/// and at each axis accept only equal sizes or a 1 on one side.
///
/// Deliberately not sharing a line of code with the implementation under test.
/// A reference that reuses the code it checks proves the code agrees with
/// itself.
fn reference(lhs: &[usize], rhs: &[usize]) -> Option<Vec<usize>> {
    let rank = lhs.len().max(rhs.len());
    let mut out = Vec::with_capacity(rank);
    for axis in 0..rank {
        let from_end = rank - axis;
        let l = lhs.len().checked_sub(from_end).map_or(1, |i| lhs[i]);
        let r = rhs.len().checked_sub(from_end).map_or(1, |i| rhs[i]);
        out.push(match (l, r) {
            (l, r) if l == r => l,
            (1, r) => r,
            (l, 1) => l,
            _ => return None,
        });
    }
    Some(out)
}

/// The rule under test, as dimensions.
fn broadcast(lhs: &[usize], rhs: &[usize]) -> Option<Vec<usize>> {
    let lhs_field = <Dyn as Shape>::from_dyn(lhs).expect("`Dyn` accepts any dimensions");
    let rhs_field = <Dyn as Shape>::from_dyn(rhs).expect("`Dyn` accepts any dimensions");
    <Dyn as BroadcastShape<Dyn>>::output_shape(&lhs_field, &rhs_field)
        .ok()
        .map(|out| Dyn::dims(&out))
}

const CASES: usize = 5000;
const MAX_RANK: usize = 6;

// --- properties ----------------------------------------------------------

#[test]
fn the_rule_agrees_with_the_reference_on_both_answers() {
    let mut rng = Rng::new(0x5148_502d_3030_37);
    let (mut accepted, mut rejected) = (0, 0);

    for case in 0..CASES {
        let lhs = rng.shape(MAX_RANK);
        let rhs = rng.shape(MAX_RANK);
        let expected = reference(&lhs, &rhs);

        assert_eq!(
            broadcast(&lhs, &rhs),
            expected,
            "case {case}: {lhs:?} against {rhs:?}"
        );

        if expected.is_some() {
            accepted += 1;
        } else {
            rejected += 1;
        }
    }

    // A generator that never produced an incompatible pair would make the
    // agreement above vacuous on half the rule.
    assert!(accepted > 100, "only {accepted} pairs broadcast");
    assert!(rejected > 100, "only {rejected} pairs were rejected");
}

#[test]
fn broadcasting_is_commutative_in_the_shape_it_produces() {
    let mut rng = Rng::new(0x436f_6d6d_75_74);

    for case in 0..CASES {
        let lhs = rng.shape(MAX_RANK);
        let rhs = rng.shape(MAX_RANK);

        assert_eq!(
            broadcast(&lhs, &rhs),
            broadcast(&rhs, &lhs),
            "case {case}: {lhs:?} against {rhs:?} depends on operand order"
        );
    }
}

#[test]
fn broadcasting_is_associative_where_all_three_are_defined() {
    let mut rng = Rng::new(0x4173_736f_6369);
    let mut reached = 0;

    for case in 0..CASES {
        let a = rng.shape(MAX_RANK);
        let b = rng.shape(MAX_RANK);
        let c = rng.shape(MAX_RANK);

        let left = broadcast(&a, &b).and_then(|ab| broadcast(&ab, &c));
        let right = broadcast(&b, &c).and_then(|bc| broadcast(&a, &bc));

        // Not an unconditional law: `[2]` and `[3]` fail together, but
        // grouping can reach an intermediate that one order rejects and the
        // other never forms. Where both orders produce an answer, the answers
        // must be the same one.
        if let (Some(left), Some(right)) = (&left, &right) {
            assert_eq!(left, right, "case {case}: {a:?}, {b:?}, {c:?}");
            reached += 1;
        }
    }

    assert!(
        reached > 100,
        "only {reached} triples were defined both ways"
    );
}

#[test]
fn a_shape_of_ones_is_the_identity() {
    let mut rng = Rng::new(0x4964_656e_74);

    for case in 0..CASES {
        let shape = rng.shape(MAX_RANK);
        let ones = vec![1; shape.len()];

        assert_eq!(
            broadcast(&shape, &ones).as_deref(),
            Some(shape.as_slice()),
            "case {case}: {shape:?} was changed by broadcasting against ones"
        );
    }
}

#[test]
fn a_scalar_is_absorbed_whatever_the_other_operand_is() {
    let mut rng = Rng::new(0x5363_616c_61);

    for case in 0..CASES {
        let shape = rng.shape(MAX_RANK);

        assert_eq!(
            broadcast(&shape, &[]).as_deref(),
            Some(shape.as_slice()),
            "case {case}: a scalar changed {shape:?}"
        );
    }
}

#[test]
fn a_result_axis_is_never_one_where_an_operand_axis_was_zero() {
    // The `max` bug this rule is written to avoid: `1` against `0` must give
    // `0`, and `max` gives `1`. Stated as a property so it is checked at every
    // axis position and rank rather than only in the one hand-written case.
    let mut rng = Rng::new(0x5a65_726f);
    let mut zeros_seen = 0;

    for case in 0..CASES {
        let lhs = rng.shape(MAX_RANK);
        let rhs = rng.shape(MAX_RANK);
        let Some(out) = broadcast(&lhs, &rhs) else {
            continue;
        };

        let rank = out.len();
        for axis in 0..rank {
            let from_end = rank - axis;
            let l = lhs.len().checked_sub(from_end).map_or(1, |i| lhs[i]);
            let r = rhs.len().checked_sub(from_end).map_or(1, |i| rhs[i]);
            if l == 0 || r == 0 {
                assert_eq!(
                    out[axis], 0,
                    "case {case}: {lhs:?} against {rhs:?} lost a zero at axis {axis}"
                );
                zeros_seen += 1;
            }
        }
    }

    assert!(zeros_seen > 100, "only {zeros_seen} zero axes were reached");
}

#[test]
fn the_result_has_the_rank_of_the_longer_operand() {
    let mut rng = Rng::new(0x5261_6e6b);

    for case in 0..CASES {
        let lhs = rng.shape(MAX_RANK);
        let rhs = rng.shape(MAX_RANK);
        let Some(out) = broadcast(&lhs, &rhs) else {
            continue;
        };

        assert_eq!(
            out.len(),
            lhs.len().max(rhs.len()),
            "case {case}: {lhs:?} against {rhs:?}"
        );
    }
}
