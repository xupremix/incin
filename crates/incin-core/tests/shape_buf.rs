//! Property and boundary tests for `ShapeBuf` / `StrideBuf` (`SHP-003`).
//!
//! The properties are checked against `u128` reference implementations. That is
//! the whole point of the exercise: the defect these types exist to prevent is
//! `usize` arithmetic silently wrapping in a release build, so a test that
//! recomputes the expectation in `usize` would wrap in exactly the same way and
//! agree with the bug.
//!
//! Randomized cases use a fixed-seed xorshift generator rather than a
//! `proptest` dependency. Every failure is reproducible from the case index
//! printed in the assertion message, and the generator deliberately biases
//! toward the values that break unchecked code: 0, 1, `usize::MAX`, and factors
//! near `2^32`.

use incin_core::prelude::{
    INLINE_RANK, OperationKind, RankExpectation, ShapeBuf, ShapeError, StrideBuf,
};

const OP: OperationKind = OperationKind::Storage;

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

    /// A dimension drawn from the values that actually break shape arithmetic,
    /// not from a uniform range where overflow essentially never happens.
    fn dim(&mut self) -> usize {
        match self.below(10) {
            0 => 0,
            1 => 1,
            2 => usize::MAX,
            3 => 1 << 32,
            4 => (1 << 32) + 1,
            5 => usize::MAX / 3,
            _ => self.below(64) as usize,
        }
    }

    fn shape(&mut self, max_rank: usize) -> Vec<usize> {
        let rank = self.below(max_rank as u64 + 1) as usize;
        (0..rank).map(|_| self.dim()).collect()
    }
}

// --- u128 reference implementations -------------------------------------

// The generator emits `usize::MAX` freely, so a product of eight dimensions can
// exceed `u128` as well. Every reference is therefore itself checked and yields
// `None` for "too large to represent" — which, for our purposes, is just a
// stronger form of "does not fit `usize`".

fn reference_numel(dims: &[usize]) -> Option<u128> {
    // The mathematical product, which is 0 whenever any factor is — not a
    // left-to-right fold, whose intermediate overflow would depend on the
    // order the axes happen to be written in.
    if dims.contains(&0) {
        return Some(0);
    }
    dims.iter()
        .try_fold(1u128, |acc, &d| acc.checked_mul(d as u128))
}

fn reference_contiguous(dims: &[usize]) -> Option<Vec<u128>> {
    let mut strides = vec![1u128; dims.len()];
    for axis in (0..dims.len().saturating_sub(1)).rev() {
        strides[axis] = strides[axis + 1].checked_mul(dims[axis + 1] as u128)?;
    }
    Some(strides)
}

fn reference_span(dims: &[usize], strides: &[usize]) -> Option<u128> {
    if dims.contains(&0) {
        return Some(0);
    }
    dims.iter().zip(strides).try_fold(1u128, |acc, (&d, &s)| {
        acc.checked_add((d as u128 - 1).checked_mul(s as u128)?)
    })
}

fn fits(value: Option<u128>) -> bool {
    value.is_some_and(|v| v <= usize::MAX as u128)
}

// --- properties ---------------------------------------------------------

#[test]
fn dims_round_trip_across_the_spill_boundary() {
    let mut rng = Rng::new(0x5EED_0001);
    for case in 0..2_000 {
        let dims = rng.shape(INLINE_RANK * 2 + 2);
        let buf = ShapeBuf::from_slice(&dims);
        assert_eq!(buf.dims(), dims.as_slice(), "case {case}");
        assert_eq!(buf.rank(), dims.len(), "case {case}");
        assert_eq!(
            buf.is_inline(),
            dims.len() <= INLINE_RANK,
            "case {case}: spill boundary moved"
        );
    }
}

#[test]
fn representation_does_not_affect_equality() {
    let mut rng = Rng::new(0x5EED_0002);
    for case in 0..2_000 {
        let dims = rng.shape(INLINE_RANK * 2 + 2);

        // Same contents reached three ways: bulk copy, repeated push (which
        // spills incrementally), and an iterator collect.
        let bulk = ShapeBuf::from_slice(&dims);
        let mut pushed = ShapeBuf::scalar();
        for &d in &dims {
            pushed.push(d);
        }
        let collected: ShapeBuf = dims.iter().copied().collect();

        assert_eq!(bulk, pushed, "case {case}");
        assert_eq!(bulk, collected, "case {case}");
        assert_eq!(bulk.dims(), pushed.dims(), "case {case}");
    }
}

#[test]
fn numel_agrees_with_u128_and_reports_overflow() {
    let mut rng = Rng::new(0x5EED_0003);
    let mut saw_overflow = false;
    let mut saw_success = false;

    for case in 0..5_000 {
        let dims = rng.shape(INLINE_RANK + 2);
        let buf = ShapeBuf::from_slice(&dims);
        let reference = reference_numel(&dims);

        match buf.checked_numel(OP) {
            Ok(n) => {
                saw_success = true;
                assert!(fits(reference), "case {case}: {dims:?} should have overflowed");
                assert_eq!(Some(n as u128), reference, "case {case}: {dims:?}");
            }
            Err(err) => {
                saw_overflow = true;
                assert!(!fits(reference), "case {case}: {dims:?} fits but errored");
                assert!(
                    matches!(err, ShapeError::ArithmeticOverflow { operation, .. } if operation == OP),
                    "case {case}: wrong error {err}"
                );
            }
        }
        assert_eq!(buf.numel().is_some(), fits(reference), "case {case}");
    }

    // A property test that only ever exercises one branch proves nothing about
    // the other; assert the generator reached both.
    assert!(saw_overflow, "generator never produced an overflowing shape");
    assert!(saw_success, "generator never produced a representable shape");
}

#[test]
fn numel_does_not_depend_on_axis_order() {
    // Multiplication is commutative; a running product with an early exit on
    // overflow is not. A shape containing both `usize::MAX` and `0` collapses
    // to 0 if the fold reaches the zero first and errors if it does not, so
    // this invariant is what forces the zero case to be short-circuited.
    let mut rng = Rng::new(0x5EED_0008);

    for case in 0..5_000 {
        let dims = rng.shape(INLINE_RANK);
        let forward = ShapeBuf::from_slice(&dims).numel();

        let mut reversed = dims.clone();
        reversed.reverse();
        assert_eq!(
            ShapeBuf::from_slice(&reversed).numel(),
            forward,
            "case {case}: {dims:?} reversed disagrees"
        );

        // And under an arbitrary rotation, not just the reversal.
        if !dims.is_empty() {
            let pivot = rng.below(dims.len() as u64) as usize;
            let mut rotated = dims[pivot..].to_vec();
            rotated.extend_from_slice(&dims[..pivot]);
            assert_eq!(
                ShapeBuf::from_slice(&rotated).numel(),
                forward,
                "case {case}: {dims:?} rotated by {pivot} disagrees"
            );
        }
    }
}

#[test]
fn an_empty_tensor_is_empty_however_large_its_other_axes_are() {
    // The regression this pins: the fold used to error on the second form.
    assert_eq!(ShapeBuf::from_slice(&[usize::MAX, 0, usize::MAX]).numel(), Some(0));
    assert_eq!(ShapeBuf::from_slice(&[usize::MAX, usize::MAX, 0]).numel(), Some(0));
    assert_eq!(
        ShapeBuf::from_slice(&[usize::MAX, usize::MAX, 0])
            .checked_byte_len(8, OP)
            .unwrap(),
        0
    );
}

#[test]
fn byte_len_overflows_separately_from_element_count() {
    let mut rng = Rng::new(0x5EED_0004);
    let mut saw_split = false;

    for case in 0..5_000 {
        let dims = rng.shape(INLINE_RANK);
        let element_size = [1usize, 2, 4, 8, 16][rng.below(5) as usize];
        let buf = ShapeBuf::from_slice(&dims);

        let numel = reference_numel(&dims);
        let bytes = numel.and_then(|n| n.checked_mul(element_size as u128));

        match buf.checked_byte_len(element_size, OP) {
            Ok(n) => {
                assert!(fits(bytes), "case {case}: {dims:?} x {element_size}");
                assert_eq!(Some(n as u128), bytes, "case {case}");
            }
            Err(_) => {
                assert!(!fits(bytes), "case {case}: {dims:?} x {element_size}");
                // The interesting case: the count fits but the byte length
                // does not. Unchecked code sizes an allocation from the count
                // and then indexes it in bytes.
                if fits(numel) {
                    saw_split = true;
                }
            }
        }
    }

    assert!(
        saw_split,
        "generator never produced a shape whose count fits but whose byte length does not"
    );
}

#[test]
fn contiguous_strides_agree_with_u128_and_report_overflow() {
    let mut rng = Rng::new(0x5EED_0005);
    let mut saw_overflow = false;

    for case in 0..5_000 {
        let dims = rng.shape(INLINE_RANK + 2);
        let shape = ShapeBuf::from_slice(&dims);
        let reference = reference_contiguous(&dims);
        let representable = reference
            .as_ref()
            .is_some_and(|s| s.iter().all(|&v| fits(Some(v))));

        match StrideBuf::contiguous_for(&shape, OP) {
            Ok(strides) => {
                assert!(representable, "case {case}: {dims:?} should have overflowed");
                assert_eq!(strides.len(), dims.len(), "case {case}");
                let reference = reference.as_ref().expect("representable");
                for (axis, (&got, &want)) in strides.strides().iter().zip(reference).enumerate() {
                    assert_eq!(got as u128, want, "case {case}: axis {axis} of {dims:?}");
                }
                assert!(strides.is_contiguous_for(&shape), "case {case}");
            }
            Err(err) => {
                saw_overflow = true;
                assert!(!representable, "case {case}: {dims:?} fits but errored");
                assert!(
                    matches!(err, ShapeError::ArithmeticOverflow { .. }),
                    "case {case}: wrong error {err}"
                );
            }
        }
    }

    assert!(saw_overflow, "generator never overflowed a stride");
}

#[test]
fn span_agrees_with_u128() {
    let mut rng = Rng::new(0x5EED_0006);

    for case in 0..5_000 {
        let dims = rng.shape(INLINE_RANK);
        let strides: Vec<usize> = (0..dims.len()).map(|_| rng.dim()).collect();
        let shape = ShapeBuf::from_slice(&dims);
        let stride_buf = StrideBuf::from_slice(&strides);
        let reference = reference_span(&dims, &strides);

        match stride_buf.checked_span(&shape, OP) {
            Ok(n) => {
                assert!(fits(reference), "case {case}: {dims:?} / {strides:?}");
                assert_eq!(
                    Some(n as u128),
                    reference,
                    "case {case}: {dims:?} / {strides:?}"
                );
            }
            Err(err) => {
                assert!(!fits(reference), "case {case}: {dims:?} / {strides:?}");
                assert!(
                    matches!(err, ShapeError::ArithmeticOverflow { .. }),
                    "case {case}: wrong error {err}"
                );
            }
        }
    }
}

#[test]
fn push_and_pop_round_trip_across_the_boundary() {
    let mut rng = Rng::new(0x5EED_0007);

    for case in 0..1_000 {
        let dims = rng.shape(INLINE_RANK * 2 + 2);
        let mut buf = ShapeBuf::scalar();
        for &d in &dims {
            buf.push(d);
        }
        for &expected in dims.iter().rev() {
            assert_eq!(buf.pop(), Some(expected), "case {case}");
        }
        assert_eq!(buf.pop(), None, "case {case}");
        assert_eq!(buf.rank(), 0, "case {case}");
    }
}

// --- boundaries ---------------------------------------------------------

#[test]
fn scalar_shape_holds_one_element() {
    let scalar = ShapeBuf::scalar();
    assert_eq!(scalar.rank(), 0);
    assert_eq!(scalar.checked_numel(OP).unwrap(), 1);
    assert_eq!(scalar.checked_byte_len(4, OP).unwrap(), 4);
    assert!(!scalar.is_empty_tensor());

    // A rank-0 view spans exactly its one element.
    let strides = StrideBuf::contiguous_for(&scalar, OP).unwrap();
    assert!(strides.is_empty());
    assert_eq!(strides.checked_span(&scalar, OP).unwrap(), 1);
}

#[test]
fn a_zero_dimension_holds_no_elements_and_spans_nothing() {
    let shape = ShapeBuf::from_slice(&[3, 0, 4]);
    assert!(shape.is_empty_tensor());
    assert_eq!(shape.checked_numel(OP).unwrap(), 0);
    assert_eq!(shape.checked_byte_len(4, OP).unwrap(), 0);

    let strides = StrideBuf::contiguous_for(&shape, OP).unwrap();
    assert_eq!(strides.strides(), &[0, 4, 1]);
    assert_eq!(strides.checked_span(&shape, OP).unwrap(), 0);
}

#[test]
fn broadcast_strides_span_less_than_the_element_count() {
    // A stride-0 axis addresses the same element repeatedly. Sizing a buffer
    // from `numel` here would over-allocate; sizing it from `span` is correct.
    let shape = ShapeBuf::from_slice(&[4, 3]);
    let strides = StrideBuf::from_slice(&[0, 1]);
    assert_eq!(shape.checked_numel(OP).unwrap(), 12);
    assert_eq!(strides.checked_span(&shape, OP).unwrap(), 3);
}

#[test]
fn span_rejects_a_stride_buffer_of_the_wrong_rank() {
    let shape = ShapeBuf::from_slice(&[2, 3, 4]);
    let strides = StrideBuf::from_slice(&[12, 4]);
    let err = strides.checked_span(&shape, OperationKind::Slice).unwrap_err();
    assert_eq!(
        err,
        ShapeError::RankMismatch {
            operation: OperationKind::Slice,
            expected: RankExpectation::SameAs {
                operand: "shape",
                rank: 3
            },
            actual: 2,
        }
    );
}

#[test]
fn spill_boundary_is_exactly_inline_rank() {
    let inline = ShapeBuf::from_slice(&vec![2; INLINE_RANK]);
    let spilled = ShapeBuf::from_slice(&vec![2; INLINE_RANK + 1]);
    assert!(inline.is_inline(), "rank {INLINE_RANK} should stay inline");
    assert!(!spilled.is_inline(), "rank {} should spill", INLINE_RANK + 1);

    // Contents survive the spill, and the two ranks stay distinguishable.
    assert_eq!(inline.dims().len(), INLINE_RANK);
    assert_eq!(spilled.dims().len(), INLINE_RANK + 1);
    assert!(spilled.dims().iter().all(|&d| d == 2));
}

#[test]
fn nothing_derived_is_cached() {
    // The RFC forbids caching a value that can fall out of step with the
    // dimensions. Mutating a dimension in place must change every derived
    // quantity immediately.
    let mut shape = ShapeBuf::from_slice(&[2, 3, 4]);
    assert_eq!(shape.checked_numel(OP).unwrap(), 24);
    assert_eq!(
        StrideBuf::contiguous_for(&shape, OP).unwrap().strides(),
        &[12, 4, 1]
    );

    shape.dims_mut()[1] = 5;
    assert_eq!(shape.checked_numel(OP).unwrap(), 40);
    assert_eq!(
        StrideBuf::contiguous_for(&shape, OP).unwrap().strides(),
        &[20, 4, 1]
    );

    shape.push(2);
    assert_eq!(shape.rank(), 4);
    assert_eq!(shape.checked_numel(OP).unwrap(), 80);
}

#[test]
fn overflow_terms_are_named_distinctly() {
    // Diagnostics have to point at one multiplication, not at the whole rule.
    let huge = ShapeBuf::from_slice(&[usize::MAX, 2]);
    let numel_err = huge.checked_numel(OperationKind::Reshape).unwrap_err();
    assert_eq!(
        numel_err.to_string(),
        "reshape: arithmetic overflow evaluating 'product of dimensions'"
    );

    let wide = ShapeBuf::from_slice(&[usize::MAX / 2]);
    let byte_err = wide.checked_byte_len(4, OperationKind::Storage).unwrap_err();
    assert_eq!(
        byte_err.to_string(),
        "storage: arithmetic overflow evaluating 'element count * element size'"
    );

    // Contiguous strides accumulate from the right, so the overflowing shape
    // here needs a large *trailing* dimension, not a large leading one.
    let deep = ShapeBuf::from_slice(&[2, usize::MAX, 4]);
    let stride_err = StrideBuf::contiguous_for(&deep, OperationKind::Permute).unwrap_err();
    assert_eq!(
        stride_err.to_string(),
        "permute: arithmetic overflow evaluating 'stride * trailing dimension'"
    );
}

#[test]
fn is_contiguous_for_is_false_on_a_transposed_view() {
    let shape = ShapeBuf::from_slice(&[3, 2]);
    assert!(StrideBuf::from_slice(&[2, 1]).is_contiguous_for(&shape));
    assert!(!StrideBuf::from_slice(&[1, 3]).is_contiguous_for(&shape));

    // A shape whose contiguous strides overflow is contiguous under nothing.
    let overflowing = ShapeBuf::from_slice(&[2, usize::MAX, usize::MAX]);
    assert!(StrideBuf::contiguous_for(&overflowing, OP).is_err());
    assert!(!StrideBuf::from_slice(&[1, 1, 1]).is_contiguous_for(&overflowing));
}
