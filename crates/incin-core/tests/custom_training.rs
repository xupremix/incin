//! Downstream proof that a custom operation can train.
//!
//! Written entirely against the public extension surface, from the perspective
//! of a user crate: a custom `square` operation (`y = x^2`) declares its
//! contract, runs its forward kernel on the CPU backend, records its backward
//! recipe through [`tape_record`](incin_backends::cpu::tape_record), and the
//! resulting gradients flow through the standard [`AutogradBackend`] backward
//! pass. A companion test proves the dtype story in the other direction: the
//! same operation advertises `f32` only, and an `f16` invocation is refused
//! with a typed reason before any kernel runs.

extern crate incin_core as incin;

use incin_backends::cpu::{CpuBackendImpl, CpuBuffer, CpuStorage, tape_record_with};
use incin_core::backend_authoring::{
    AutogradBackend, DescriptorError, DifferentiableOp, LogicalTensorMeta, Operation,
    OperationIdentity, OperationKey, SupportLevel, TapeStorage, TensorId,
};
use incin_core::exec::catalog::NoAttributes;
use incin_core::exec::{CanonicalError, ExecutionContext, GradMode, TensorHandle};
use incin_core::prelude::{BackendError, Cpu, DTypeId, ErrorMessage, Local, OperationKind, f16};

/// `y = x^2`, elementwise, `f32` only. The operation a user would write for a
/// custom activation or loss term: one input, one output, same shape.
#[derive(Debug, Clone)]
struct Square;

impl Operation for Square {
    type Attributes = NoAttributes;

    const KEY: OperationKey = OperationKey {
        namespace: std::borrow::Cow::Borrowed("company.example"),
        name: std::borrow::Cow::Borrowed("square"),
        version: 1,
    };

    fn infer_outputs(
        _attributes: &Self::Attributes,
        inputs: &[LogicalTensorMeta],
    ) -> Result<Vec<LogicalTensorMeta>, DescriptorError> {
        Ok(inputs.first().cloned().into_iter().collect())
    }
}

/// Allocate an `f32` output, preserving a construction failure inside a
/// structured backend error rather than relabelling it as an input refusal:
// the inputs passed inference, so a failure here is the kernel's, not theirs.
fn contiguous_f32(values: Vec<f32>, dims: &[usize]) -> Result<CpuStorage, BackendError> {
    CpuStorage::try_from_contiguous(CpuBuffer::F32(values), dims).map_err(|error| {
        BackendError::Execution {
            operation: OperationKind::Pointwise,
            message: ErrorMessage::new(error.to_string()),
        }
    })
}

impl DifferentiableOp<CpuBackendImpl<Cpu>> for Square {
    type Dtype = f32;
    /// The saved input itself: the recipe needs every `x` next to its gradient.
    type Saved = CpuStorage;

    fn supports(query: &incin_core::exec::CapabilityQuery) -> SupportLevel {
        assert_eq!(query.operation, OperationIdentity::Custom(Square::KEY));
        // Layer 3 of the dtype contract: this kernel is `f32` only, so say so
        // per query. Anything else is refused before launch, never executed
        // against a dtype it was not written for.
        if query.dtype != DTypeId::F32.descriptor() {
            SupportLevel::Unsupported(incin_core::exec::UnsupportedReason::CustomOperation {
                operation: Square::KEY,
            })
        } else {
            SupportLevel::Native
        }
    }

    fn forward(
        inputs: &[CpuStorage],
        _attributes: &NoAttributes,
    ) -> Result<(CpuStorage, Self::Saved), BackendError> {
        let x = inputs.first().cloned().ok_or(BackendError::InvalidInput {
            operation: OperationKind::Pointwise,
            reason: "square requires one CPU input",
        })?;
        // Layer 4 of the dtype contract: the descriptor already promised
        // `f32` (see `supports`), and the kernel proves the buffer agrees
        // rather than trusting the advertisement.
        if x.metadata().dtype != DTypeId::F32.descriptor() {
            return Err(BackendError::InvalidInput {
                operation: OperationKind::Pointwise,
                reason: "square kernel holds f32 only",
            });
        }
        let dims = x.metadata().shape.dims().to_vec();
        let flat: usize = dims.iter().product();
        let mut values = Vec::with_capacity(flat);
        // Rank-agnostic read via the public scalar accessor: `get` returns the
        // value as `f64` for any numeric buffer, so no buffer matching needed.
        let mut index = vec![0usize; dims.len()];
        for _ in 0..flat {
            let v = x.get(&index) as f32;
            values.push(v * v);
            odometer(&mut index, &dims);
        }
        let out = contiguous_f32(values, &dims)?;
        Ok((out, x))
    }

    fn backward(
        saved: &CpuStorage,
        grad_out: &CpuStorage,
    ) -> incin_core::error::Result<Vec<CpuStorage>> {
        // dy/dx = 2x, with `x` owned by the recipe rather than borrowed from
        // the live graph.
        let dims = saved.metadata().shape.dims().to_vec();
        let flat: usize = dims.iter().product();
        let mut grads = Vec::with_capacity(flat);
        let mut index = vec![0usize; dims.len()];
        for _ in 0..flat {
            grads.push(2.0f32 * saved.get(&index) as f32 * grad_out.get(&index) as f32);
            odometer(&mut index, &dims);
        }
        Ok(vec![contiguous_f32_for_recipe(grads, &dims)?])
    }
}

/// `contiguous_f32` for a backward recipe, whose error type is the core error
/// rather than the backend error.
fn contiguous_f32_for_recipe(
    values: Vec<f32>,
    dims: &[usize],
) -> Result<CpuStorage, incin_core::error::Error> {
    CpuStorage::try_from_contiguous(CpuBuffer::F32(values), dims)
}

/// Odometer-style row-major multi-index increment.
fn odometer(index: &mut [usize], shape: &[usize]) {
    for i in (0..index.len()).rev() {
        index[i] += 1;
        if index[i] < shape[i] {
            return;
        }
        index[i] = 0;
    }
}

fn square_forward(ctx: &ExecutionContext<CpuBackendImpl<Cpu>>, x: &CpuStorage) -> CpuStorage {
    let handle = TensorHandle::from_storage::<CpuBackendImpl<Cpu>, f32, Local>(x);
    incin_core::backend_authoring::execute::<Square, _>(ctx, NoAttributes, &[handle])
        .expect("square executes on f32 CPU input")
}

#[test]
fn downstream_custom_operation_trains_end_to_end() {
    let ctx = ExecutionContext::new(CpuBackendImpl::<Cpu>::new());
    let x =
        CpuStorage::try_from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]), vec![4]).unwrap();
    let x_id: TensorId = x.id();

    // Forward: y = x^2.
    let loss = square_forward(&ctx, &x);
    assert_eq!(loss.metadata().shape.dims(), &[4]);
    for (i, expected) in [1.0f64, 4.0, 9.0, 16.0].iter().enumerate() {
        assert!(
            (loss.get(&[i]) - expected).abs() < 1e-6,
            "forward mismatch at {i}"
        );
    }

    // Finite-difference cross-check of the recipe, run under NoGrad so the
    // probe executions leave no nodes behind for the real backward pass.
    let analytic = 2.0 * 3.0f64;
    let eps = 1e-2f64;
    let (plus, minus) = GradMode::Disabled.scope(|| {
        let xp = CpuStorage::try_from_contiguous(
            CpuBuffer::F32(vec![1.0, 2.0, 3.0 + eps as f32, 4.0]),
            vec![4],
        )
        .unwrap();
        let xm = CpuStorage::try_from_contiguous(
            CpuBuffer::F32(vec![1.0, 2.0, 3.0 - eps as f32, 4.0]),
            vec![4],
        )
        .unwrap();
        (
            square_forward(&ctx, &xp).get(&[2]),
            square_forward(&ctx, &xm).get(&[2]),
        )
    });
    let numeric = (plus - minus) / (2.0 * eps);
    assert!(
        (numeric - analytic).abs() < 5e-2,
        "finite-difference {numeric} disagrees with analytic {analytic}"
    );

    // Backward through the standard backend pass: the custom node drains from
    // the same tape the built-in kernels use.
    let grads =
        <CpuBackendImpl<Cpu> as AutogradBackend>::backward::<f32>(&loss).expect("backward runs");
    let gx = grads.get(x_id).expect("custom input receives a gradient");
    for (i, expected) in [2.0f64, 4.0, 6.0, 8.0].iter().enumerate() {
        assert!(
            (gx.get(&[i]) - expected).abs() < 1e-5,
            "gradient mismatch at {i}"
        );
    }
}

#[test]
fn downstream_custom_operation_records_nothing_under_no_grad() {
    use incin_backends::cpu::tape_depth;

    let ctx = ExecutionContext::new(CpuBackendImpl::<Cpu>::new());
    let x = CpuStorage::try_from_contiguous(CpuBuffer::F32(vec![1.0, 2.0]), vec![2]).unwrap();
    let before = tape_depth();
    GradMode::Disabled.scope(|| {
        square_forward(&ctx, &x);
        // The lazy form constructs nothing either: the closure must not run.
        tape_record_with(|| panic!("record_with built an entry under NoGrad"));
    });
    assert_eq!(
        tape_depth(),
        before,
        "a NoGrad custom forward must record nothing"
    );
}

#[test]
fn downstream_custom_operation_refuses_dtypes_it_does_not_support() {
    let ctx = ExecutionContext::new(CpuBackendImpl::<Cpu>::new());
    let x = CpuStorage::try_from_contiguous(
        CpuBuffer::F16(vec![f16::from_f32(1.0), f16::from_f32(2.0)]),
        vec![2],
    )
    .unwrap();
    let handle = TensorHandle::from_storage::<CpuBackendImpl<Cpu>, f16, Local>(&x);
    let err = incin_core::backend_authoring::execute::<Square, _>(&ctx, NoAttributes, &[handle])
        .expect_err("f16 must be refused before any kernel runs");
    match err {
        CanonicalError::Backend(BackendError::Unsupported { reason, .. }) => {
            assert_eq!(
                reason,
                incin_core::exec::UnsupportedReason::CustomOperation {
                    operation: Square::KEY,
                }
            );
        }
        other => panic!("expected a typed Unsupported refusal, got {other:?}"),
    }
}
