//! The value half of the CPU-oracle conformance harness (issue #83).
//!
//! Issue #83 makes the CPU backend the oracle for every advertised capability
//! row. Two suites together deliver that claim:
//!
//! * `tests/conformance_oracle.rs` checks that every advertised tuple
//!   *executes* (and that unadvertised dtypes are refused, and that training
//!   rows record something on the tape). It cannot check values, because its
//!   oracle and its subject are the same backend - a comparison of a backend
//!   against itself is a comparison of a value against its own bits.
//! * This suite checks that the values are *right*. Every case runs the CPU
//!   backend in `f32` and compares the result element-wise against an
//!   independent oracle written here as `f64` scalar loops. The loops
//!   re-derive each operation from its definition rather than calling any
//!   framework compute path, so the oracle shares no arithmetic with the
//!   implementation under test - an oracle that did could not catch a bug the
//!   two of them share.
//!
//! # What a case is
//!
//! Table-driven: `(case id, operation, shapes, fixed inputs)` executed on the
//! CPU backend through the canonical descriptor dispatch, compared against
//! the oracle under the single tolerance table below. Inputs are fixed
//! literals plus one fixed-seed generated block, so a failure reproduces
//! exactly. A mismatch reports the operation, the case id, and the first
//! offending element with its tolerance and the tolerance's recorded reason;
//! one run reports *every* failing case rather than stopping at the first,
//! and a kernel panic in one case is a failure for that case rather than the
//! end of the run.
//!
//! # What this suite does NOT yet cover (honestly)
//!
//! * **Gradients, the `Training: yes` column, gradcheck, autograd.** Forward
//!   values only. Checking a derivative needs finite differences or a
//!   recorded reference, which is a separate harness - the issue lists
//!   training rows as their own acceptance criterion.
//! * **Any backend other than CPU.** The issue's cross-backend comparison
//!   (CPU as oracle, CUDA/WGPU as subject) needs the #82 runner; nothing
//!   here executes on a device.
//! * **Dtypes other than `f32`.** No f64/f16/bf16/integer/quantized
//!   operands; the oracle arithmetic is written for `f32` inputs.
//! * **Layouts other than contiguous.** One broadcast *shape* pair is posed,
//!   but every operand is physically contiguous - no strided, transposed, or
//!   sliced views.
//! * **Ranks beyond the table.** Inputs are rank 1 and 2; the scalar
//!   reductions exercise rank-0 *outputs*. Zero-element tensors, rank-0
//!   inputs, and rank above two are not posed, nor are NaN/Inf inputs.
//! * **Registry enumeration.** The case table is hand-written, so it can
//!   drift from `docs/capabilities.md`. `src/conformance` already enumerates
//!   advertised tuples from the registry for the execution half; driving
//!   *values* from the same enumeration is the intended next step, and is
//!   what makes a newly advertised row automatically value-tested.
//! * **Machine-readable artifacts.** No `target/conformance/*.json` yet; the
//!   report is a test-failure string.
//! * **Unadvertised-tuple refusals and `ImplementationKind` dispatch.**
//!   Checked by `tests/conformance_oracle.rs`, not repeated here.
//!
//! # Why this lives in tests/ and not in `src/conformance`
//!
//! Tests-only, and no public hook was needed: canonical dispatch,
//! `TensorHandle`, and `CpuStorage` already expose everything the harness
//! touches. Keeping the oracle out of `src/` is a correctness property, not
//! a packaging preference - the moment the reference implementation shares a
//! module with the backend it judges, the two can drift into agreeing while
//! both are wrong. The issue's proposed `Conformance::new().oracle()...
//! .run()` runner shape remains the target for the cross-backend slice,
//! where a second backend has to be plugged in as the subject.

#![cfg(feature = "cpu")]

use incin_backends::cpu::{CpuBackendImpl, CpuBuffer, CpuStorage};
use incin_core::exec::catalog::{AxisAttributes, NoAttributes, op};
use incin_core::exec::{ExecutionContext, GradMode, TensorHandle, dispatch};
use incin_core::shapes::OperationKind;

// ============================================================================
// The one tolerance table
// ============================================================================

/// How far one operation's `f32` result may land from the `f64` oracle.
///
/// A value passes if it is within *either* bound: an absolute bound alone
/// rejects large values that are correct to every bit a float has, and a
/// relative bound alone rejects values near zero, where the reference has no
/// magnitude to be relative to.
#[derive(Debug, Clone, Copy)]
struct Tolerance {
    /// Absolute error allowed before failure, in value units.
    absolute: f64,
    /// Relative error allowed before failure, as a fraction of magnitude.
    relative: f64,
    /// Why the bound is what it is. Required for every row: a tolerance
    /// without a recorded reason is how one operation quietly grows a looser
    /// bar than its neighbours.
    reason: &'static str,
}

impl Tolerance {
    /// Whether `actual` is an acceptable answer for `expected`.
    fn accepts(self, expected: f64, actual: f64) -> bool {
        if expected == actual {
            return true;
        }
        if !expected.is_finite() || !actual.is_finite() {
            // Matching magnitudes of non-finite values says nothing; only
            // bit-level agreement (handled by the equality above) does.
            return false;
        }
        let difference = (expected - actual).abs();
        difference <= self.absolute || difference <= self.relative * expected.abs()
    }
}

/// The single tolerance table for this harness.
///
/// Every operation the case table poses appears here exactly once - the
/// coverage test below fails if the two lists ever disagree - and every row
/// carries the reason for its bound. The issue's rule is "tolerances live in
/// one table, and every exception carries a recorded reason"; this is that
/// table, in miniature. When a second backend joins the harness, this table
/// moves to shared test support rather than being copied per suite.
const TOLERANCES: &[(OperationKind, Tolerance)] = &[
    (
        OperationKind::Add,
        Tolerance {
            absolute: 1e-6,
            relative: 1e-5,
            reason: "f32 addition is correctly rounded against an exact f64 sum; \
                     the absolute half covers cancellation to near zero",
        },
    ),
    (
        OperationKind::Sub,
        Tolerance {
            absolute: 1e-6,
            relative: 1e-5,
            reason: "f32 subtraction is correctly rounded against an exact f64 \
                     difference; the absolute half covers cancellation to near zero",
        },
    ),
    (
        OperationKind::Mul,
        Tolerance {
            absolute: 1e-6,
            relative: 1e-5,
            reason: "one correctly rounded f32 multiply differs from the exact \
                     product by half an ulp",
        },
    ),
    (
        OperationKind::Div,
        Tolerance {
            absolute: 1e-6,
            relative: 1e-5,
            reason: "one correctly rounded f32 divide differs from the exact \
                     quotient by half an ulp; case divisors stay away from zero \
                     so the quotient stays finite",
        },
    ),
    (
        OperationKind::Neg,
        Tolerance {
            absolute: 0.0,
            relative: 0.0,
            reason: "negation flips a sign bit and performs no arithmetic; any \
                     difference means the kernel computed something it was not asked to",
        },
    ),
    (
        OperationKind::Abs,
        Tolerance {
            absolute: 0.0,
            relative: 0.0,
            reason: "abs discards a sign bit and performs no arithmetic; any \
                     difference means the kernel computed something it was not asked to",
        },
    ),
    (
        OperationKind::Relu,
        Tolerance {
            absolute: 0.0,
            relative: 0.0,
            reason: "relu selects its input or zero; both are exactly \
                     representable, so the result must be bit-identical",
        },
    ),
    (
        OperationKind::Exp,
        Tolerance {
            absolute: 1e-5,
            relative: 1e-5,
            reason: "the f32 exp kernel against an f64 libm exp; a correctly \
                     rounded f32 exp differs by under one ulp, and the looser \
                     absolute half absorbs any polynomial evaluation path",
        },
    ),
    (
        OperationKind::SumAll,
        Tolerance {
            absolute: 1e-3,
            relative: 1e-4,
            reason: "f32 sequential accumulation over up to 64 elements against \
                     an exact f64 sum; the worst case scales as n * eps * sum|x| \
                     and lands near 1e-4 for the seeded case",
        },
    ),
    (
        OperationKind::SumDim,
        Tolerance {
            absolute: 1e-5,
            relative: 1e-5,
            reason: "three- and four-element f32 accumulations; about an order \
                     of magnitude above the pointwise bound because a short sum \
                     of opposing signs can land near zero",
        },
    ),
    (
        OperationKind::MeanDim,
        Tolerance {
            absolute: 1e-5,
            relative: 1e-5,
            reason: "a short f32 accumulation followed by one division by an \
                     extent (3) that is not a power of two, so the division \
                     rounds on top of the sum's error",
        },
    ),
    (
        OperationKind::MaxAll,
        Tolerance {
            absolute: 0.0,
            relative: 0.0,
            reason: "max selects an input value and performs no arithmetic; the \
                     oracle reads the same inputs, so the results must match bit \
                     for bit",
        },
    ),
];

/// The recorded bound for one operation. Loud rather than defaulting: an
/// operation posed by the case table with no tolerance row is a harness bug,
/// and a silent default would let a new operation skip the review its bound
/// deserves.
fn tolerance_for(operation: OperationKind) -> Tolerance {
    TOLERANCES
        .iter()
        .find(|(candidate, _)| *candidate == operation)
        .map(|(_, tolerance)| *tolerance)
        .unwrap_or_else(|| {
            panic!(
                "no tolerance recorded for {operation}; every operation the case \
                 table poses belongs in TOLERANCES with a recorded reason"
            )
        })
}

// ============================================================================
// The case table
// ============================================================================

/// One operand of a case: a shape and the row-major `f32` values the backend
/// will receive.
struct Operand {
    shape: Vec<usize>,
    values: Vec<f32>,
}

/// One table-driven case: which operation, over which fixed inputs.
struct Case {
    /// Stable identifier reported on mismatch.
    id: &'static str,
    /// The operation, used for dispatch, tolerance lookup, and reporting.
    operation: OperationKind,
    /// The reduction axis for `*Dim` operations; `None` otherwise.
    axis: Option<usize>,
    /// Operands in catalog arity order.
    inputs: Vec<Operand>,
}

/// Build one case from borrowed shapes and values, copying into owned form
/// so the one generated case can borrow its buffer from the stack above.
fn row(
    id: &'static str,
    operation: OperationKind,
    axis: Option<usize>,
    inputs: Vec<(&[usize], &[f32])>,
) -> Case {
    Case {
        id,
        operation,
        axis,
        inputs: inputs
            .into_iter()
            .map(|(shape, values)| Operand {
                shape: shape.to_vec(),
                values: values.to_vec(),
            })
            .collect(),
    }
}

/// Fixed seed for the generated accumulation case. Integer-only LCG
/// arithmetic, so the values are identical on every platform and every run -
/// a failure reproduces from the seed without anyone pasting a buffer.
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// `count` deterministic values in `[-1, 1)` from `seed`.
fn seeded_values(seed: u64, count: usize) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Top 24 bits -> [0, 1) in f64, then stretch to [-1, 1).
            let unit = f64::from((state >> 40) as u32) / (1u64 << 24) as f64;
            (unit * 2.0 - 1.0) as f32
        })
        .collect()
}

/// The case table: 15 cases over 12 operations. Adding a case is data only;
/// adding an operation also requires its row in `TOLERANCES`.
fn table() -> Vec<Case> {
    let seeded = seeded_values(SEED, 64);
    vec![
        row(
            "add_elementwise",
            OperationKind::Add,
            None,
            vec![
                (&[2, 3], &[1.0, -2.5, 3.25, 0.5, 0.0, -4.75]),
                (&[2, 3], &[10.0, 0.2, -3.25, 0.5, -1.0, 4.75]),
            ],
        ),
        row(
            "add_broadcast_row",
            OperationKind::Add,
            None,
            vec![
                (&[2, 3], &[0.1, 0.2, 0.3, 1.1, 2.3, -1.7]),
                (&[3], &[1.0, -2.0, 0.25]),
            ],
        ),
        row(
            "sub_mixed_signs",
            OperationKind::Sub,
            None,
            vec![
                (&[2, 3], &[5.5, -0.25, 2.75, -8.0, 0.125, 3.5]),
                (&[2, 3], &[1.25, 0.25, -2.75, 2.0, 0.125, -7.25]),
            ],
        ),
        row(
            "mul_zeros_and_signs",
            OperationKind::Mul,
            None,
            vec![
                (&[2, 3], &[0.0, -1.5, 2.5, -0.4, 3.25, 1.1]),
                (&[2, 3], &[7.5, 0.0, -2.5, 4.0, -1.5, 0.5]),
            ],
        ),
        row(
            "div_away_from_zero",
            OperationKind::Div,
            None,
            vec![
                (&[2, 3], &[1.5, -0.3, 2.75, 10.0, -0.1, 0.0]),
                (&[2, 3], &[2.0, 0.5, 0.25, 4.0, -1.5, 8.0]),
            ],
        ),
        row(
            "neg_values",
            OperationKind::Neg,
            None,
            vec![(&[2, 3], &[1.5, -0.25, 0.0, 3.75, -2.5, 0.125])],
        ),
        row(
            "abs_mixed_signs",
            OperationKind::Abs,
            None,
            vec![(&[2, 3], &[-1.5, 0.25, 0.0, -3.75, 2.5, -0.125])],
        ),
        row(
            "relu_at_the_kink",
            OperationKind::Relu,
            None,
            vec![(&[4], &[-3.5, 0.0, 2.25, -0.125])],
        ),
        row(
            "exp_controlled_range",
            OperationKind::Exp,
            None,
            vec![(&[2, 3], &[-5.0, -1.25, 0.0, 0.75, 2.5, 5.0])],
        ),
        row(
            "sum_all_small",
            OperationKind::SumAll,
            None,
            vec![(
                &[3, 4],
                &[
                    1.0, -2.5, 3.25, 0.5, //
                    -4.75, 0.1, 2.3, -1.9, //
                    0.75, -0.4, 5.5, -3.25,
                ],
            )],
        ),
        row(
            "sum_all_seeded_sixty_four",
            OperationKind::SumAll,
            None,
            vec![(&[4, 16], seeded.as_slice())],
        ),
        row(
            "sum_dim_axis_zero",
            OperationKind::SumDim,
            Some(0),
            vec![(
                &[3, 4],
                &[
                    1.0, -2.5, 3.25, 0.5, //
                    -4.75, 0.1, 2.3, -1.9, //
                    0.75, -0.4, 5.5, -3.25,
                ],
            )],
        ),
        row(
            "sum_dim_axis_one",
            OperationKind::SumDim,
            Some(1),
            vec![(
                &[3, 4],
                &[
                    1.0, -2.5, 3.25, 0.5, //
                    -4.75, 0.1, 2.3, -1.9, //
                    0.75, -0.4, 5.5, -3.25,
                ],
            )],
        ),
        row(
            "mean_dim_axis_zero",
            OperationKind::MeanDim,
            Some(0),
            vec![(
                &[3, 4],
                &[
                    1.0, -2.5, 3.25, 0.5, //
                    -4.75, 0.1, 2.3, -1.9, //
                    0.75, -0.4, 5.5, -3.25,
                ],
            )],
        ),
        row(
            "max_all_small",
            OperationKind::MaxAll,
            None,
            vec![(
                &[3, 4],
                &[
                    1.0, -2.5, 3.25, 0.5, //
                    -4.75, 0.1, 2.3, -1.9, //
                    0.75, -0.4, 5.5, -3.25,
                ],
            )],
        ),
    ]
}

/// Ratchet: the table must not shrink without a deliberate edit here. Raise
/// it when cases land; lowering it means cases were removed, which should be
/// a visible decision rather than a quiet one.
const CASE_FLOOR: usize = 15;

// ============================================================================
// The oracle: f64 scalar loops, written from each operation's definition
// ============================================================================

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Row-major flat position of a multi-index.
fn raveled(index: &[usize], shape: &[usize]) -> usize {
    index
        .iter()
        .zip(shape)
        .fold(0, |acc, (&i, &extent)| acc * extent + i)
}

/// Multi-index at a row-major flat position.
fn unraveled(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut index = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        index[axis] = flat % shape[axis];
        flat /= shape[axis];
    }
    index
}

/// NumPy-style right-aligned broadcasting of two operand shapes.
fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>, String> {
    let rank = a.len().max(b.len());
    let mut out = Vec::with_capacity(rank);
    for axis in 0..rank {
        let da = if axis + a.len() >= rank {
            a[axis + a.len() - rank]
        } else {
            1
        };
        let db = if axis + b.len() >= rank {
            b[axis + b.len() - rank]
        } else {
            1
        };
        if da != db && da != 1 && db != 1 {
            return Err(format!(
                "shapes {a:?} and {b:?} do not broadcast against each other"
            ));
        }
        out.push(da.max(db));
    }
    Ok(out)
}

/// Map one output multi-index onto an operand's own multi-index under
/// broadcasting (extent-1 axes read position 0).
fn operand_index(out: &[usize], shape: &[usize]) -> Vec<usize> {
    let offset = out.len() - shape.len();
    shape
        .iter()
        .enumerate()
        .map(|(axis, &extent)| if extent == 1 { 0 } else { out[offset + axis] })
        .collect()
}

/// The f64 arithmetic each binary operation denotes.
fn binary_fn(operation: OperationKind) -> Option<fn(f64, f64) -> f64> {
    match operation {
        OperationKind::Add => Some(|a, b| a + b),
        OperationKind::Sub => Some(|a, b| a - b),
        OperationKind::Mul => Some(|a, b| a * b),
        OperationKind::Div => Some(|a, b| a / b),
        _ => None,
    }
}

/// The f64 arithmetic each unary operation denotes.
fn unary_fn(operation: OperationKind) -> Option<fn(f64) -> f64> {
    match operation {
        OperationKind::Neg => Some(|a| -a),
        OperationKind::Abs => Some(f64::abs),
        OperationKind::Relu => Some(|a| a.max(0.0)),
        OperationKind::Exp => Some(f64::exp),
        _ => None,
    }
}

fn expect_operand(case: &Case, position: usize) -> Result<&Operand, String> {
    case.inputs.get(position).ok_or_else(|| {
        format!(
            "the oracle expected at least {} operands and the table posed {}",
            position + 1,
            case.inputs.len()
        )
    })
}

fn expect_lengths(operand: &Operand) -> Result<(), String> {
    let expected = numel(&operand.shape);
    if operand.values.len() != expected {
        return Err(format!(
            "the table itself is inconsistent: shape {:?} needs {expected} values, \
             the case carries {}",
            operand.shape,
            operand.values.len()
        ));
    }
    Ok(())
}

fn binary_oracle(
    case: &Case,
    op_fn: fn(f64, f64) -> f64,
) -> Result<(Vec<usize>, Vec<f64>), String> {
    let lhs = expect_operand(case, 0)?;
    let rhs = expect_operand(case, 1)?;
    expect_lengths(lhs)?;
    expect_lengths(rhs)?;
    let out_shape = broadcast_shapes(&lhs.shape, &rhs.shape)?;
    let out_count = numel(&out_shape);
    let mut out = Vec::with_capacity(out_count);
    for flat in 0..out_count {
        let index = unraveled(flat, &out_shape);
        let a = f64::from(lhs.values[raveled(&operand_index(&index, &lhs.shape), &lhs.shape)]);
        let b = f64::from(rhs.values[raveled(&operand_index(&index, &rhs.shape), &rhs.shape)]);
        out.push(op_fn(a, b));
    }
    Ok((out_shape, out))
}

fn unary_oracle(case: &Case, op_fn: fn(f64) -> f64) -> Result<(Vec<usize>, Vec<f64>), String> {
    let input = expect_operand(case, 0)?;
    expect_lengths(input)?;
    let out = input
        .values
        .iter()
        .map(|&value| op_fn(f64::from(value)))
        .collect();
    Ok((input.shape.clone(), out))
}

fn reduce_all_oracle(case: &Case) -> Result<(Vec<usize>, Vec<f64>), String> {
    let input = expect_operand(case, 0)?;
    expect_lengths(input)?;
    if input.values.is_empty() {
        return Err("zero-element reductions are outside this suite's scope".to_string());
    }
    let value = match case.operation {
        OperationKind::SumAll => input.values.iter().map(|&v| f64::from(v)).sum(),
        OperationKind::MeanAll => {
            let sum: f64 = input.values.iter().map(|&v| f64::from(v)).sum();
            sum / input.values.len() as f64
        }
        OperationKind::MaxAll => input
            .values
            .iter()
            .map(|&v| f64::from(v))
            .fold(f64::NEG_INFINITY, f64::max),
        other => return Err(format!("the oracle has no *All rule for {other}")),
    };
    Ok((Vec::new(), vec![value]))
}

fn reduce_axis_oracle(case: &Case) -> Result<(Vec<usize>, Vec<f64>), String> {
    let input = expect_operand(case, 0)?;
    expect_lengths(input)?;
    let axis = case.axis.ok_or_else(|| {
        format!(
            "{} is a dim reduction but the case carries no axis",
            case.id
        )
    })?;
    if axis >= input.shape.len() {
        return Err(format!(
            "axis {axis} is outside the operand's rank {}",
            input.shape.len()
        ));
    }
    let extent = input.shape[axis];
    if extent == 0 {
        return Err("zero-element reductions are outside this suite's scope".to_string());
    }
    let mut out_shape = input.shape.clone();
    out_shape.remove(axis);
    let out_count = numel(&out_shape);
    let mut out = Vec::with_capacity(out_count);
    for flat in 0..out_count {
        let out_index = unraveled(flat, &out_shape);
        let mut total = 0.0_f64;
        for step in 0..extent {
            let mut in_index = out_index.clone();
            in_index.insert(axis, step);
            total += f64::from(input.values[raveled(&in_index, &input.shape)]);
        }
        let value = match case.operation {
            OperationKind::SumDim => total,
            OperationKind::MeanDim => total / extent as f64,
            OperationKind::MaxDim => {
                // Recompute as a max over the same indices; `total` above is
                // a sum, which max does not use.
                let mut best = f64::NEG_INFINITY;
                for step in 0..extent {
                    let mut in_index = out_index.clone();
                    in_index.insert(axis, step);
                    best = best.max(f64::from(input.values[raveled(&in_index, &input.shape)]));
                }
                best
            }
            other => return Err(format!("the oracle has no axis rule for {other}")),
        };
        out.push(value);
    }
    Ok((out_shape, out))
}

/// Run the oracle for one case: expected output shape and expected values.
fn oracle(case: &Case) -> Result<(Vec<usize>, Vec<f64>), String> {
    if let Some(op_fn) = binary_fn(case.operation) {
        return binary_oracle(case, op_fn);
    }
    if let Some(op_fn) = unary_fn(case.operation) {
        return unary_oracle(case, op_fn);
    }
    match case.operation {
        OperationKind::SumAll | OperationKind::MeanAll | OperationKind::MaxAll => {
            reduce_all_oracle(case)
        }
        OperationKind::SumDim | OperationKind::MeanDim | OperationKind::MaxDim => {
            reduce_axis_oracle(case)
        }
        other => Err(format!("the oracle does not know how to compute {other}")),
    }
}

// ============================================================================
// Execution on the CPU backend
// ============================================================================

type Backend = CpuBackendImpl;

fn execute_case(
    context: &ExecutionContext<Backend>,
    case: &Case,
    inputs: &[CpuStorage],
) -> Result<CpuStorage, String> {
    let handles: Vec<TensorHandle<'_>> = inputs
        .iter()
        .map(TensorHandle::from_storage::<Backend, f32, _>)
        .collect();
    let axis = case.axis;
    let result = match case.operation {
        OperationKind::Add => dispatch::execute::<op::Add, _>(context, NoAttributes, &handles),
        OperationKind::Sub => dispatch::execute::<op::Sub, _>(context, NoAttributes, &handles),
        OperationKind::Mul => dispatch::execute::<op::Mul, _>(context, NoAttributes, &handles),
        OperationKind::Div => dispatch::execute::<op::Div, _>(context, NoAttributes, &handles),
        OperationKind::Neg => dispatch::execute::<op::Neg, _>(context, NoAttributes, &handles),
        OperationKind::Abs => dispatch::execute::<op::Abs, _>(context, NoAttributes, &handles),
        OperationKind::Relu => dispatch::execute::<op::Relu, _>(context, NoAttributes, &handles),
        OperationKind::Exp => dispatch::execute::<op::Exp, _>(context, NoAttributes, &handles),
        OperationKind::SumAll => {
            dispatch::execute::<op::SumAll, _>(context, NoAttributes, &handles)
        }
        OperationKind::MaxAll => {
            dispatch::execute::<op::MaxAll, _>(context, NoAttributes, &handles)
        }
        OperationKind::SumDim => {
            let axis = axis.ok_or_else(|| format!("{} carries no axis", case.id))?;
            dispatch::execute::<op::SumDim, _>(context, AxisAttributes { axis }, &handles)
        }
        OperationKind::MeanDim => {
            let axis = axis.ok_or_else(|| format!("{} carries no axis", case.id))?;
            dispatch::execute::<op::MeanDim, _>(context, AxisAttributes { axis }, &handles)
        }
        other => return Err(format!("the harness has no dispatcher for {other}")),
    };
    result.map_err(|error| format!("{error}"))
}

/// Read every value of a result storage, plus the shape it reports.
fn read_storage(storage: &CpuStorage) -> (Vec<usize>, Vec<f64>) {
    let shape = storage.shape().dims().to_vec();
    let count = numel(&shape);
    let values = (0..count)
        .map(|flat| storage.get(&unraveled(flat, &shape)))
        .collect();
    (shape, values)
}

// ============================================================================
// The runner
// ============================================================================

/// One failed case, kept structured so the report can name it precisely.
struct Failure {
    case: &'static str,
    operation: OperationKind,
    detail: String,
}

impl Failure {
    fn render(&self) -> String {
        format!("FAIL {} ({}): {}", self.case, self.operation, self.detail)
    }
}

/// Shape and value comparison for one executed case.
fn compare(
    case: &Case,
    actual_shape: &[usize],
    actual: &[f64],
    expected_shape: &[usize],
    expected: &[f64],
) -> Result<(), String> {
    if actual_shape != expected_shape {
        return Err(format!(
            "output shape is {actual_shape:?} but the oracle derives {expected_shape:?}"
        ));
    }
    if actual.len() != expected.len() {
        return Err(format!(
            "read back {} values but the oracle derives {}",
            actual.len(),
            expected.len()
        ));
    }
    let tolerance = tolerance_for(case.operation);
    for (index, (&want, &got)) in expected.iter().zip(actual).enumerate() {
        if !tolerance.accepts(want, got) {
            return Err(format!(
                "element {index} is {got}, expected {want}, outside tolerance \
                 abs={} rel={} ({})",
                tolerance.absolute, tolerance.relative, tolerance.reason
            ));
        }
    }
    Ok(())
}

/// Run every case once and collect every failure. Never stops early: a
/// reader wants all findings from one run, not the first.
fn run() -> Vec<Failure> {
    let context = ExecutionContext::new(Backend::new());
    let mut failures = Vec::new();

    for case in table() {
        // A panic is a finding for this case, not the end of the run: the
        // descriptor contract says a bad invocation comes back as a value,
        // and a harness that dies on the first panic reports one case where
        // a reader wanted all of them.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // Forward values only: keep the ambient gradient recording off so
            // this suite cannot grow a tape it never reads, and so the mode
            // is set by name rather than inherited from a default that could
            // change underneath the measurement.
            GradMode::Disabled.scope(|| -> Result<(), String> {
                let storages: Vec<CpuStorage> = case
                    .inputs
                    .iter()
                    .map(|operand| {
                        expect_lengths(operand)?;
                        CpuStorage::try_from_contiguous(
                            CpuBuffer::F32(operand.values.clone()),
                            &operand.shape,
                        )
                        .map_err(|error| {
                            format!(
                                "the harness could not build {:?} storage: {error}",
                                operand.shape
                            )
                        })
                    })
                    .collect::<Result<Vec<CpuStorage>, String>>()?;
                let output = execute_case(&context, &case, &storages)?;
                let (actual_shape, actual) = read_storage(&output);
                let (expected_shape, expected) = oracle(&case)?;
                compare(&case, &actual_shape, &actual, &expected_shape, &expected)
            })
        }));
        match outcome {
            Ok(Ok(())) => {}
            Ok(Err(detail)) => failures.push(Failure {
                case: case.id,
                operation: case.operation,
                detail,
            }),
            Err(_) => failures.push(Failure {
                case: case.id,
                operation: case.operation,
                detail: "panicked; the descriptor contract requires a returned \
                         error, and a panic hides whatever the value would \
                         have been"
                    .to_string(),
            }),
        }
    }
    failures
}

// ============================================================================
// The tests
// ============================================================================

#[test]
fn cpu_values_match_the_f64_scalar_oracle() {
    let failures = run();
    assert!(
        failures.is_empty(),
        "cpu backend vs f64 scalar-loop oracle: {} of {} case(s) failed:\n  {}",
        failures.len(),
        table().len(),
        failures
            .iter()
            .map(Failure::render)
            .collect::<Vec<_>>()
            .join("\n  ")
    );
}

/// The tolerance table and the case table are two views of one harness; if
/// they disagree, either an operation runs with no reviewed bound or a bound
/// is recorded for nothing. Both are drift the issue's "one table" rule
/// exists to prevent.
#[test]
fn tolerance_table_matches_the_case_table() {
    let mut from_cases: Vec<OperationKind> =
        table().into_iter().map(|case| case.operation).collect();
    from_cases.sort_unstable();
    from_cases.dedup();

    let mut from_tolerances: Vec<OperationKind> =
        TOLERANCES.iter().map(|(operation, _)| *operation).collect();
    from_tolerances.sort_unstable();
    let deduplicated = from_tolerances.len();
    from_tolerances.dedup();
    assert_eq!(
        deduplicated,
        from_tolerances.len(),
        "TOLERANCES lists an operation more than once: {:?}",
        TOLERANCES
            .iter()
            .map(|(operation, _)| *operation)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        from_cases, from_tolerances,
        "the case table and the tolerance table must pose exactly the same \
         operations"
    );

    for (operation, tolerance) in TOLERANCES {
        assert!(
            !tolerance.reason.is_empty(),
            "{operation} has a tolerance with no recorded reason"
        );
        assert!(
            tolerance.absolute >= 0.0 && tolerance.relative >= 0.0,
            "{operation} has a negative tolerance bound"
        );
    }
}

#[test]
fn the_case_table_does_not_shrink() {
    let cases = table();
    assert!(
        cases.len() >= CASE_FLOOR,
        "the conformance value table fell to {} cases, below the recorded floor \
         of {CASE_FLOOR}",
        cases.len()
    );
    let mut ids: Vec<&str> = cases.iter().map(|case| case.id).collect();
    ids.sort_unstable();
    let unique = ids.len();
    ids.dedup();
    assert_eq!(
        unique,
        ids.len(),
        "case ids must be unique so a failure names exactly one case"
    );
}
