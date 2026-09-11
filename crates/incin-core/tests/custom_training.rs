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
        _attributes: &NoAttributes,
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

/// A chain that mixes built-in operations with a custom one walks as one graph.
///
/// This is the property the extension seam exists for, and the one a
/// downstream crate could not have before `tape_record` was public: the custom
/// operation's node lands on the same thread-local tape the built-in kernels
/// record on, so a single backward call crosses all of them. Before, a
/// downstream author owned a separate node list, and a graph with a built-in
/// operation on either side of their kernel came apart at the seam without
/// saying so.
///
/// The chain is `sum((x * x)^2)`, so the closed form is `sum(x^4)` and the
/// gradient is `4 x^3`. Built-in `Mul` below the custom operation, built-in
/// `SumAll` above it: if either half failed to join, the gradient would be
/// absent rather than wrong, which is why this asserts values and not just
/// presence.
#[test]
fn a_graph_mixing_builtin_and_custom_operations_walks_as_one() {
    use incin_core::exec::catalog::op;

    let ctx = ExecutionContext::new(CpuBackendImpl::<Cpu>::new());
    let values = [1.0f32, 2.0, 3.0, 4.0];
    let x = CpuStorage::try_from_contiguous(CpuBuffer::F32(values.to_vec()), vec![4]).unwrap();
    let x_id: TensorId = x.id();

    fn handle(storage: &CpuStorage) -> TensorHandle<'_> {
        TensorHandle::from_storage::<CpuBackendImpl<Cpu>, f32, Local>(storage)
    }

    // Built-in, below the custom operation.
    let squared = incin_core::backend_authoring::execute::<op::Mul, _>(
        &ctx,
        NoAttributes,
        &[handle(&x), handle(&x)],
    )
    .expect("the built-in multiply runs");

    // The custom operation, in the middle.
    let quartic = square_forward(&ctx, &squared);

    // Built-in, above it.
    let loss = incin_core::backend_authoring::execute::<op::SumAll, _>(
        &ctx,
        NoAttributes,
        &[handle(&quartic)],
    )
    .expect("the built-in reduction runs");

    let grads =
        <CpuBackendImpl<Cpu> as AutogradBackend>::backward::<f32>(&loss).expect("backward runs");
    let gx = grads
        .get(x_id)
        .expect("the gradient crossed both built-in operations and the custom one");

    for (i, value) in values.iter().enumerate() {
        let expected = 4.0 * f64::from(*value).powi(3);
        assert!(
            (gx.get(&[i]) - expected).abs() < 1e-3,
            "d/dx sum(x^4) at {i}: got {}, want {expected}",
            gx.get(&[i])
        );
    }
}

/// The recipe, swept against central differences by the public checker.
///
/// This is the check a custom-operation author should run before trusting a
/// backward rule, and it is one call: `gradcheck` perturbs every element of
/// every input, re-runs the forward under `NoGrad` so the probes leave
/// nothing on the tape, and compares each slope against what the recipe
/// produced at that element.
///
/// `for_f32()` carries the step size, which is the part that is easy to get
/// wrong alone: at `f32` precision the total error is minimised near `1e-2`,
/// and the `1e-4` that looks conservative sits at its own noise floor.
///
/// The closure reduces to a scalar, because one central difference
/// approximates the whole gradient contribution of the element it perturbed
/// only for a scalar output. Reduce the way the model does.
#[test]
fn the_public_gradcheck_sweeps_the_custom_recipe() {
    use incin_core::exec::catalog::op;
    use incin_core::exec::{GradCheckOptions, gradcheck};

    let ctx = ExecutionContext::new(CpuBackendImpl::<Cpu>::new());
    let x =
        CpuStorage::try_from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]), vec![4]).unwrap();

    let scalar_loss = |inputs: &[CpuStorage]| -> incin_core::error::Result<CpuStorage> {
        let squared = square_forward(&ctx, &inputs[0]);
        let handle = TensorHandle::from_storage::<CpuBackendImpl<Cpu>, f32, Local>(&squared);
        incin_core::backend_authoring::execute::<op::SumAll, _>(&ctx, NoAttributes, &[handle])
            .map_err(Into::into)
    };

    let report = gradcheck(scalar_loss, &[x], GradCheckOptions::for_f32())
        .expect("the sweep ran to completion");
    assert!(report.passed(), "{report}");
    assert_eq!(report.compared, 4);
}

/// The whole flow from a tensor, in one call.
///
/// This is what a custom operation should cost its author at the call site:
/// `apply_op`, then ordinary tensor operations, then `backward`. No handle,
/// no execution context, no `try_from_storage` restating the shape, dtype,
/// device and gradient marker that were already known.
///
/// `Square` is the same implementation the other tests use. Nothing about it
/// changes to be callable this way, which is the point: the trait describes
/// the operation, and how it is reached is a separate question.
#[test]
fn a_custom_operation_is_one_call_from_a_tensor() {
    use incin_core::shapes::{Dyn, ShapeBuf};
    use incin_core::tensor::base::Tensor;
    use incin_core::tensor::device::Device;
    use incin_core::tensor::grad::{Grad, RequiresGrad};

    let storage =
        CpuStorage::try_from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]), vec![4]).unwrap();
    let x = Tensor::<Dyn, CpuBackendImpl<Cpu>, f32, Grad>::try_from_storage(
        storage,
        ShapeBuf::from_slice(&[4]),
        core::marker::PhantomData,
        <Cpu as Device>::init(()),
        <Grad as RequiresGrad>::init(()),
    )
    .expect("the leaf builds");

    // The custom operation, then a built-in elementwise op, then a built-in
    // reduction, then backward. The `relu` in the middle is not decoration: it
    // is the layout-sensitive successor, and it is here so the `Dyn` layout
    // `apply_op` returns is proven to compose rather than assumed to.
    let squared = x.apply_op::<Square>(NoAttributes).expect("square runs");
    let gated = squared
        .relu()
        .expect("a built-in successor accepts the result");
    let loss = gated.sum_all().expect("the built-in reduction runs");
    let grads = loss.backward().expect("backward runs");

    let gx = grads
        .require(&x)
        .expect("the custom node is on the same graph as the built-in one");

    // Every square is positive, so the relu is the identity here and
    // d/dx sum(relu(x^2)) = 2x.
    let values: Vec<f32> = gx.to_vec1().expect("gradient reads back");
    for (i, got) in values.iter().enumerate() {
        let expected = 2.0 * (i as f32 + 1.0);
        assert!(
            (got - expected).abs() < 1e-5,
            "at {i}: got {got}, want {expected}"
        );
    }
}
