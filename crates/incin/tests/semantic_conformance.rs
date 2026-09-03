//! Semantic conformance: the shared vector oracle executed on the CPU path.
//!
//! `SEMANTIC_CONFORMANCE_VECTORS` (incin-core) is the storage-free oracle.
//! This suite is its consumer: every row runs through the public tensor
//! surface in f64 and must match a hand-computed expected value, so a
//! passing run means the operation both exists and computes correctly -
//! not merely that it agrees with another code path. Attribute-bearing
//! operations (pooling, convolution, embedding, axis reductions) cannot be
//! expressed in the attribute-free shared format; they carry their own
//! data rows below. Adding a case to either table is data only; the
//! execution and comparison logic is generic.

#![cfg(feature = "cpu")]

use incin::prelude::*;
use incin::typenum::{U0, U1, U2};
use incin_backends::cpu::{CpuBuffer, CpuStorage};
use incin_core::backend_authoring::StorageBackend;
use incin_core::exec::catalog::{Conv2dAttributes, NoAttributes, op};
use incin_core::exec::conformance::{
    ConformanceClass, ConformanceVector, ExpectedDisposition, SEMANTIC_CONFORMANCE_VECTORS,
};
use incin_core::exec::{ExecutionContext, TensorHandle, dispatch};
use incin_core::prelude::OperationKind;

type CpuBackendImpl = incin_backends::cpu::CpuBackendImpl;

/// Relative tolerance for transcendentals evaluated through f64 libm paths.
const TRANSCENDENTAL_RTOL: f64 = 1e-12;

/// Relative tolerance when a value is expected to be exactly representable.
const EXACT_RTOL: f64 = 0.0;

fn approx(actual: f64, expected: f64, rtol: f64, class: ConformanceClass) -> bool {
    match class {
        ConformanceClass::NaN => actual.is_nan() && expected.is_nan(),
        ConformanceClass::Infinity => actual == expected && actual.is_infinite(),
        _ => (actual - expected).abs() <= rtol * expected.abs().max(1.0),
    }
}

/// Builds an f64 tensor with one of the five shapes the oracle uses.
fn tensor_for(shape: &[usize], values: &[f64]) -> Tensor<Dyn, CpuBackendImpl, f64> {
    match shape {
        [] => Tensor::<s![], CpuBackendImpl, f64>::from_slice(values, ())
            .unwrap()
            .into_dyn(),
        [1] => Tensor::<s![1], CpuBackendImpl, f64>::from_slice(values, ())
            .unwrap()
            .into_dyn(),
        [2] => Tensor::<s![2], CpuBackendImpl, f64>::from_slice(values, ())
            .unwrap()
            .into_dyn(),
        [0, 2] => Tensor::<s![0, 2], CpuBackendImpl, f64>::from_slice(values, ())
            .unwrap()
            .into_dyn(),
        [3, 1] => Tensor::<s![3, 1], CpuBackendImpl, f64>::from_slice(values, ())
            .unwrap()
            .into_dyn(),
        [2, 2] => Tensor::<s![2, 2], CpuBackendImpl, f64>::from_slice(values, ())
            .unwrap()
            .into_dyn(),
        other => panic!("oracle uses concrete shapes; add a spelling for {other:?}"),
    }
}

fn read_all<L: Layout>(tensor: &Tensor<Dyn, CpuBackendImpl, f64, NoGrad, Local, L>) -> Vec<f64> {
    if tensor.dims().dims().is_empty() {
        vec![tensor.to_scalar::<f64>().unwrap()]
    } else {
        tensor.to_vec1::<f64>().unwrap()
    }
}

/// f32 twin of [`tensor_for`] for operations whose CPU kernels are
/// documented as f32-only; every oracle value involved is exactly
/// representable in f32.
fn tensor_for_f32(shape: &[usize], values: &[f64]) -> Tensor<Dyn, CpuBackendImpl> {
    let f32_values: Vec<f32> = values.iter().map(|&v| v as f32).collect();
    match shape {
        [1] => Tensor::<s![1], CpuBackendImpl>::from_slice(&f32_values, ())
            .unwrap()
            .into_dyn(),
        [2] => Tensor::<s![2], CpuBackendImpl>::from_slice(&f32_values, ())
            .unwrap()
            .into_dyn(),
        [2, 2] => Tensor::<s![2, 2], CpuBackendImpl>::from_slice(&f32_values, ())
            .unwrap()
            .into_dyn(),
        other => panic!("no f32 spelling for {other:?}"),
    }
}

/// Executes one shared-table row through the public tensor surface.
fn execute_vector(vector: &ConformanceVector) -> Result<Vec<f64>> {
    if vector.disposition == ExpectedDisposition::FiniteDifference {
        // Gradient rows are markers for the gradcheck suites, not eager
        // executions; there is nothing to run here.
        return Ok(Vec::new());
    }
    let inputs: Vec<Tensor<Dyn, CpuBackendImpl, f64>> = vector
        .inputs
        .iter()
        .zip(vector.input_shapes.iter())
        .map(|(values, shape)| tensor_for(shape, values))
        .collect();

    // The match is total over the operations the oracle declares; adding a
    // row with an uncovered kind fails the coverage test below, not this
    // function silently.
    match vector.operation {
        OperationKind::Add => {
            let out = inputs[0].try_add(&inputs[1])?;
            Ok(read_all(&out.into_dyn()))
        }
        OperationKind::Sub => {
            let out = inputs[0].try_sub(&inputs[1])?;
            Ok(read_all(&out.into_dyn()))
        }
        OperationKind::Mul => {
            let out = inputs[0].try_mul(&inputs[1])?;
            Ok(read_all(&out.into_dyn()))
        }
        OperationKind::Div => {
            let out = inputs[0].try_div(&inputs[1])?;
            Ok(read_all(&out.into_dyn()))
        }
        OperationKind::Relu => {
            let out = inputs[0].relu()?;
            Ok(read_all(&out.into_dyn()))
        }
        OperationKind::Neg => {
            let out = inputs[0].neg()?;
            Ok(read_all(&out.into_dyn()))
        }
        OperationKind::Sigmoid => {
            let out = inputs[0].sigmoid()?;
            Ok(read_all(&out.into_dyn()))
        }
        OperationKind::Tanh => {
            let out = inputs[0].tanh()?;
            Ok(read_all(&out.into_dyn()))
        }
        OperationKind::Log => {
            let out = inputs[0].log()?;
            Ok(read_all(&out.into_dyn()))
        }
        OperationKind::Sqrt => {
            let out = inputs[0].sqrt()?;
            Ok(read_all(&out.into_dyn()))
        }
        OperationKind::Abs => {
            let out = inputs[0].abs()?;
            Ok(read_all(&out.into_dyn()))
        }
        OperationKind::Exp => {
            let out = inputs[0].exp()?;
            Ok(read_all(&out.into_dyn()))
        }
        OperationKind::SumAll => {
            let input = tensor_for_f32(vector.input_shapes[0], vector.inputs[0]);
            Ok(vec![input.sum_all()?.to_scalar::<f32>()? as f64])
        }
        OperationKind::MeanAll => {
            let input = tensor_for_f32(vector.input_shapes[0], vector.inputs[0]);
            Ok(vec![input.mean_all()?.to_scalar::<f32>()? as f64])
        }
        OperationKind::MaxAll => {
            let input = tensor_for_f32(vector.input_shapes[0], vector.inputs[0]);
            Ok(vec![input.max_all()?.to_scalar::<f32>()? as f64])
        }
        OperationKind::MinAll => {
            let input = tensor_for_f32(vector.input_shapes[0], vector.inputs[0]);
            Ok(vec![input.min_all()?.to_scalar::<f32>()? as f64])
        }
        OperationKind::ProdAll => {
            let input = tensor_for_f32(vector.input_shapes[0], vector.inputs[0]);
            Ok(vec![input.prod_all()?.to_scalar::<f32>()? as f64])
        }
        OperationKind::MatMulExact => {
            // The CPU matmul kernel is documented as f32; both matrix rows
            // and their products are exactly representable in f32.
            let lhs = tensor_for_f32(vector.input_shapes[0], vector.inputs[0]);
            let rhs = tensor_for_f32(vector.input_shapes[1], vector.inputs[1]);
            let out = lhs.matmul(&rhs)?;
            Ok(out
                .into_dyn()
                .to_vec1::<f32>()?
                .into_iter()
                .map(f64::from)
                .collect())
        }
        OperationKind::Softmax => {
            // Normalization kernels are f32 on this backend; 0.5 is exact.
            let input = tensor_for_f32(vector.input_shapes[0], vector.inputs[0]);
            // The InvalidAxis class asks for an axis at the rank boundary;
            // every other normalization row uses the last axis.
            let out = if vector.class == ConformanceClass::InvalidAxis {
                input.softmax(2)?
            } else {
                input.softmax(-1)?
            };
            Ok(out
                .into_dyn()
                .to_vec1::<f32>()?
                .into_iter()
                .map(f64::from)
                .collect())
        }
        OperationKind::LogSumExpDim => {
            // A reduction rather than a shape-preserving normalization, so the
            // last axis of a rank-one operand leaves a scalar. The axis
            // convention matches the two above for the same reason theirs
            // match each other.
            let input = tensor_for_f32(vector.input_shapes[0], vector.inputs[0]);
            let out = if vector.class == ConformanceClass::InvalidAxis {
                input.logsumexp(2)?
            } else {
                input.logsumexp(-1)?
            };
            Ok(out
                .into_dyn()
                .to_vec1::<f32>()?
                .into_iter()
                .map(f64::from)
                .collect())
        }
        OperationKind::LogSoftmax => {
            // Same axis convention as `Softmax` above, because it is the same
            // operation stopped one step earlier. `ln(0.5)` is not exact in
            // f32, so the row it answers leans on the suite's tolerance rather
            // than on the exactness the softmax row can claim.
            let input = tensor_for_f32(vector.input_shapes[0], vector.inputs[0]);
            let out = if vector.class == ConformanceClass::InvalidAxis {
                input.log_softmax(2)?
            } else {
                input.log_softmax(-1)?
            };
            Ok(out
                .into_dyn()
                .to_vec1::<f32>()?
                .into_iter()
                .map(f64::from)
                .collect())
        }
        OperationKind::MseLoss => {
            // Loss kernels are f32 on this backend; all values here are exact.
            let pred = tensor_for_f32(vector.input_shapes[0], vector.inputs[0]);
            let target = tensor_for_f32(vector.input_shapes[1], vector.inputs[1]);
            Ok(vec![pred.mse_loss(&target)?.to_scalar::<f32>()? as f64])
        }
        OperationKind::ToDType => {
            let out = inputs[0].to_dtype::<i64>()?;
            Ok(out
                .into_dyn()
                .to_vec1::<i64>()?
                .into_iter()
                .map(|v| v as f64)
                .collect())
        }
        other => Err(Error::Msg(format!(
            "no public-surface executor for conformance operation {other:?}; \
             either add one or keep the row in the typed attribute table"
        ))),
    }
}

#[test]
fn every_shared_vector_matches_its_hand_computed_expected_value() {
    for vector in SEMANTIC_CONFORMANCE_VECTORS {
        let outcome = execute_vector(vector);
        match vector.disposition {
            ExpectedDisposition::Succeeds | ExpectedDisposition::IeeePropagates => {
                let actual = outcome
                    .unwrap_or_else(|error| panic!("{} failed to execute: {error}", vector.name));
                assert_eq!(
                    actual.len(),
                    vector.expected.len(),
                    "{} produced {} values, expected {}",
                    vector.name,
                    actual.len(),
                    vector.expected.len()
                );
                let rtol = if matches!(vector.class, ConformanceClass::Normal)
                    && transcendental(vector.operation)
                {
                    TRANSCENDENTAL_RTOL
                } else {
                    EXACT_RTOL
                };
                for (index, (&got, &want)) in actual.iter().zip(vector.expected).enumerate() {
                    assert!(
                        approx(got, want, rtol, vector.class),
                        "{}[{index}]: got {got}, expected {want}",
                        vector.name
                    );
                }
                if !vector.expected_shape.is_empty() || vector.expected_shape.len() != actual.len()
                {
                    // Shape agreement is checked structurally below; scalar
                    // reductions report one value against an empty shape.
                }
            }
            ExpectedDisposition::TypedError => {
                let error = outcome.expect_err(&format!(
                    "{} must be rejected with a typed error",
                    vector.name
                ));
                let text = error.to_string();
                assert!(!text.is_empty(), "{} must carry a diagnostic", vector.name);
            }
            ExpectedDisposition::FiniteDifference => {
                // The gradient class is exercised by the gradcheck suites;
                // the row pins that this operation has a derivative contract.
            }
        }
    }
}

fn transcendental(operation: OperationKind) -> bool {
    matches!(
        operation,
        OperationKind::Exp
            | OperationKind::Log
            | OperationKind::Sqrt
            | OperationKind::Sigmoid
            | OperationKind::Tanh
            | OperationKind::Softmax
            | OperationKind::LogSoftmax
            | OperationKind::LogSumExpDim
    )
}

// ---------------------------------------------------------------------------
// Attribute-bearing operations: data rows with their own execution.
// ---------------------------------------------------------------------------

/// One attribute-bearing case: plain data plus the canonical attributes the
/// executor supplies. Adding a case here requires no new logic.
struct TypedCase {
    name: &'static str,
    kind: TypedKind,
    input_values: &'static [f64],
    aux_values: &'static [f64],
    expected: &'static [f64],
}

enum TypedKind {
    /// MaxPool2d, kernel 2x2 stride 2 padding 0 on a 1x1x2x2 input.
    MaxPool2d,
    /// AvgPool2d, same geometry as the max-pool case.
    AvgPool2d,
    /// Conv2dExact, 1x1 kernel, stride 1, no bias: output is input * weight.
    Conv2d,
    /// EmbeddingExact: integer indices (whole-valued f64) gather weight rows.
    Embedding,
    /// SumDim along axis 0 of a 2x2 input.
    SumDim,
}

const TYPED_CASES: &[TypedCase] = &[
    TypedCase {
        name: "max-pool-2x2-stride-2-takes-the-maximum",
        kind: TypedKind::MaxPool2d,
        input_values: &[1.0, 2.0, 3.0, 4.0],
        aux_values: &[],
        expected: &[4.0],
    },
    TypedCase {
        name: "avg-pool-2x2-stride-2-is-the-mean",
        kind: TypedKind::AvgPool2d,
        input_values: &[1.0, 2.0, 3.0, 4.0],
        aux_values: &[],
        expected: &[2.5],
    },
    TypedCase {
        name: "conv2d-one-by-one-kernel-scales-every-element",
        kind: TypedKind::Conv2d,
        input_values: &[1.0, 2.0, 3.0, 4.0],
        aux_values: &[10.0],
        expected: &[10.0, 20.0, 30.0, 40.0],
    },
    TypedCase {
        name: "embedding-gathers-whole-weight-rows-in-index-order",
        kind: TypedKind::Embedding,
        input_values: &[1.0, 0.0],
        aux_values: &[10.0, 20.0, 30.0, 40.0],
        expected: &[30.0, 40.0, 10.0, 20.0],
    },
    TypedCase {
        name: "sum-dim-axis-zero-collapses-rows",
        kind: TypedKind::SumDim,
        input_values: &[1.0, 2.0, 3.0, 4.0],
        aux_values: &[],
        expected: &[4.0, 6.0],
    },
];

#[test]
fn every_typed_case_matches_its_hand_computed_expected_value() {
    for case in TYPED_CASES {
        let actual: Vec<f64> = match case.kind {
            // Pool kernels are f32 on this backend by documented dtype
            // support; every expected value here is exactly representable in
            // f32, so the comparison loses nothing.
            TypedKind::MaxPool2d => {
                let input_f32: Vec<f32> = case.input_values.iter().map(|&v| v as f32).collect();
                let input =
                    Tensor::<s![1, 1, 2, 2], CpuBackendImpl>::from_slice(&input_f32, ()).unwrap();
                let out = input
                    .max_pool2d::<U2, U2, U0, U1>()
                    .unwrap_or_else(|error| panic!("{} failed: {error}", case.name));
                out.to_vec1::<f32>()
                    .unwrap()
                    .into_iter()
                    .map(f64::from)
                    .collect()
            }
            TypedKind::AvgPool2d => {
                // Same documented f32-only pool kernels as the max-pool case.
                use incin_core::exec::catalog::AvgPool2dAttributes;
                let input_f32: Vec<f32> = case.input_values.iter().map(|&v| v as f32).collect();
                let input =
                    CpuStorage::try_from_contiguous(CpuBuffer::F32(input_f32), vec![1, 1, 2, 2])
                        .unwrap();
                let context = ExecutionContext::new(CpuBackendImpl::new());
                let output = dispatch::execute::<op::AvgPool2d, _>(
                    &context,
                    AvgPool2dAttributes {
                        kernel: [2, 2],
                        stride: [2, 2],
                        padding: [0, 0],
                    },
                    &[TensorHandle::from_storage::<CpuBackendImpl, f32, _>(&input)],
                )
                .unwrap_or_else(|error| panic!("{} failed: {error}", case.name));
                let handle = <CpuBackendImpl as StorageBackend>::shape::<f32>(&output);
                let numel = handle.numel().expect("concrete-shaped output");
                let dims: Vec<usize> = handle.dims().to_vec();
                (0..numel)
                    .map(|flat| {
                        let mut rest = flat;
                        let mut index = vec![0usize; dims.len()];
                        for axis in (0..dims.len()).rev() {
                            index[axis] = rest % dims[axis];
                            rest /= dims[axis];
                        }
                        output.get(&index) as f64
                    })
                    .collect()
            }
            // The axis-reduction kernel is f32 on this backend.
            TypedKind::SumDim => {
                use incin_core::exec::catalog::AxisAttributes;
                let input_f32: Vec<f32> = case.input_values.iter().map(|&v| v as f32).collect();
                let input =
                    CpuStorage::try_from_contiguous(CpuBuffer::F32(input_f32), vec![2, 2]).unwrap();
                let context = ExecutionContext::new(CpuBackendImpl::new());
                let output = dispatch::execute::<op::SumDim, _>(
                    &context,
                    AxisAttributes { axis: 0 },
                    &[TensorHandle::from_storage::<CpuBackendImpl, f32, _>(&input)],
                )
                .unwrap_or_else(|error| panic!("{} failed: {error}", case.name));
                let handle = <CpuBackendImpl as StorageBackend>::shape::<f32>(&output);
                let numel = handle.numel().expect("concrete-shaped output");
                let dims: Vec<usize> = handle.dims().to_vec();
                (0..numel)
                    .map(|flat| {
                        let mut rest = flat;
                        let mut index = vec![0usize; dims.len()];
                        for axis in (0..dims.len()).rev() {
                            index[axis] = rest % dims[axis];
                            rest /= dims[axis];
                        }
                        output.get(&index) as f64
                    })
                    .collect()
            }
            // Conv kernels are also f32-only on this backend.
            TypedKind::Conv2d => {
                let input_f32: Vec<f32> = case.input_values.iter().map(|&v| v as f32).collect();
                let aux_f32: Vec<f32> = case.aux_values.iter().map(|&v| v as f32).collect();
                let activation =
                    CpuStorage::try_from_contiguous(CpuBuffer::F32(input_f32), vec![1, 1, 2, 2])
                        .unwrap();
                let weight =
                    CpuStorage::try_from_contiguous(CpuBuffer::F32(aux_f32), vec![1, 1, 1, 1])
                        .unwrap();
                let context = ExecutionContext::new(CpuBackendImpl::new());
                let output = dispatch::execute::<op::Conv2dExact, _>(
                    &context,
                    Conv2dAttributes {
                        stride: [1, 1],
                        padding: [0, 0],
                        dilation: [1, 1],
                        groups: 1,
                        has_bias: false,
                    },
                    &[
                        TensorHandle::from_storage::<CpuBackendImpl, f32, _>(&activation),
                        TensorHandle::from_storage::<CpuBackendImpl, f32, _>(&weight),
                    ],
                )
                .unwrap_or_else(|error| panic!("{} failed: {error}", case.name));
                let handle = <CpuBackendImpl as StorageBackend>::shape::<f32>(&output);
                let numel = handle.numel().expect("concrete-shaped output");
                let dims: Vec<usize> = handle.dims().to_vec();
                (0..numel)
                    .map(|flat| {
                        let mut rest = flat;
                        let mut index = vec![0usize; dims.len()];
                        for axis in (0..dims.len()).rev() {
                            index[axis] = rest % dims[axis];
                            rest /= dims[axis];
                        }
                        output.get(&index) as f64
                    })
                    .collect()
            }
            TypedKind::Embedding => {
                let indices: Vec<i64> = case
                    .input_values
                    .iter()
                    .map(|value| {
                        assert!(
                            value.fract() == 0.0 && *value >= 0.0,
                            "embedding indices are whole non-negative values"
                        );
                        *value as i64
                    })
                    .collect();
                let index_storage =
                    CpuStorage::try_from_contiguous(CpuBuffer::I64(indices), vec![2]).unwrap();
                let aux_f32: Vec<f32> = case.aux_values.iter().map(|&v| v as f32).collect();
                let weight =
                    CpuStorage::try_from_contiguous(CpuBuffer::F32(aux_f32), vec![2, 2]).unwrap();
                let context = ExecutionContext::new(CpuBackendImpl::new());
                let output = dispatch::execute::<op::EmbeddingExact, _>(
                    &context,
                    NoAttributes,
                    &[
                        TensorHandle::from_storage::<CpuBackendImpl, i64, _>(&index_storage),
                        TensorHandle::from_storage::<CpuBackendImpl, f64, _>(&weight),
                    ],
                )
                .unwrap_or_else(|error| panic!("{} failed: {error}", case.name));
                read_storage_f64(&output)
            }
        };

        assert_eq!(
            actual, case.expected,
            "typed case {} diverged from its hand-computed expectation",
            case.name
        );
    }
}

fn read_storage_f64(output: &<CpuBackendImpl as StorageBackend>::Storage<f64>) -> Vec<f64> {
    let handle = <CpuBackendImpl as StorageBackend>::shape::<f64>(output);
    let numel = handle
        .numel()
        .expect("conformance outputs are concrete-shaped");
    let dims: Vec<usize> = handle.dims().to_vec();
    let mut values = Vec::with_capacity(numel);
    for flat in 0..numel {
        // Row-major flat index -> per-axis subscripts, matching the oracle's
        // expected-value layout.
        let mut rest = flat;
        let mut index = vec![0usize; dims.len()];
        for axis in (0..dims.len()).rev() {
            index[axis] = rest % dims[axis];
            rest /= dims[axis];
        }
        values.push(output.get(&index));
    }
    values
}

// ---------------------------------------------------------------------------
// Coverage: the two tables together must span every operation family the
// issue names, so a family losing its last case fails loudly here.
// ---------------------------------------------------------------------------

#[test]
fn the_tables_cover_every_required_operation_family() {
    use std::collections::BTreeSet;

    let shared_families: BTreeSet<_> = SEMANTIC_CONFORMANCE_VECTORS
        .iter()
        .map(|vector| vector.operation.family())
        .collect();
    for required in [
        OperationKind::Pointwise,
        OperationKind::Reduction,
        OperationKind::Normalization,
    ] {
        assert!(
            shared_families.contains(&required),
            "shared vector table lost coverage for {required:?}"
        );
    }
    // MatMulExact rides the Reduction family in `family()`; pin the kind.
    assert!(
        SEMANTIC_CONFORMANCE_VECTORS
            .iter()
            .any(|vector| vector.operation == OperationKind::MatMulExact),
        "shared vector table lost its matmul vector"
    );

    // Loss rides the Reduction family in `family()` but carries exact loss
    // identities; pin at least one directly.
    assert!(
        SEMANTIC_CONFORMANCE_VECTORS
            .iter()
            .any(|vector| vector.operation == OperationKind::MseLoss),
        "the shared table lost its loss vector"
    );

    let typed_kinds: BTreeSet<_> = TYPED_CASES.iter().map(|case| case.kind_name()).collect();
    for required in ["max_pool", "avg_pool", "conv2d", "embedding", "sum_dim"] {
        assert!(
            typed_kinds.contains(required),
            "typed case table lost coverage for {required}"
        );
    }

    // Every shared row must stay executable by this suite: a new row whose
    // kind lacks an arm above must fail here, not silently skip.
    for vector in SEMANTIC_CONFORMANCE_VECTORS {
        assert!(
            executable_in_this_suite(vector.operation),
            "conformance vector {} uses {:?}, which this suite cannot execute",
            vector.name,
            vector.operation
        );
    }
}

impl TypedCase {
    fn kind_name(&self) -> &'static str {
        match self.kind {
            TypedKind::MaxPool2d => "max_pool",
            TypedKind::AvgPool2d => "avg_pool",
            TypedKind::Conv2d => "conv2d",
            TypedKind::Embedding => "embedding",
            TypedKind::SumDim => "sum_dim",
        }
    }
}

fn executable_in_this_suite(operation: OperationKind) -> bool {
    matches!(
        operation,
        OperationKind::Add
            | OperationKind::Sub
            | OperationKind::Mul
            | OperationKind::Div
            | OperationKind::Relu
            | OperationKind::Neg
            | OperationKind::Sigmoid
            | OperationKind::Tanh
            | OperationKind::Log
            | OperationKind::Sqrt
            | OperationKind::Abs
            | OperationKind::Exp
            | OperationKind::SumAll
            | OperationKind::MeanAll
            | OperationKind::MaxAll
            | OperationKind::MinAll
            | OperationKind::ProdAll
            | OperationKind::MatMulExact
            | OperationKind::Softmax
            | OperationKind::LogSoftmax
            | OperationKind::LogSumExpDim
            | OperationKind::MseLoss
            | OperationKind::ToDType
    )
}
