//! A custom operation whose kernel is built out of built-in operations.
//!
//! The three fixtures beside this one (`custom_training.rs` here,
//! `cuda_custom_training.rs` and `wgpu_custom_training.rs` in `incin-backends`)
//! each write `y = x^2` as a hand loop over one concrete buffer variant, which
//! is why the same operation appears three times. This file writes the
//! recipe as dispatched built-in operations instead, so one implementation
//! covers every backend that can multiply and every float dtype it supports.
//!
//! That composition is only sound because the blanket `Execute` impl runs
//! `DifferentiableOp::forward` under `GradMode::Disabled`. Without that, the
//! built-in operations inside the kernel record nodes of their own against the
//! same output the custom node is recorded for, both recipes run against the
//! same output gradient, and every input receives its gradient twice. The
//! third test pins the number rather than the mechanism, so removing the
//! disable fails it.

extern crate incin_core as incin;

use core::marker::PhantomData;

use incin_backends::cpu::{CpuBackendImpl, CpuBuffer, CpuStorage};
use incin_core::backend_authoring::{
    Backend, DescriptorError, DifferentiableOp, ExecuteInto, LogicalTensorMeta, Operation,
    OperationKey, RecordingBackend, StorageOutput, SupportsDType, TapeStorage,
};
use incin_core::exec::catalog::{
    AxisAttributes, NarrowAttributes, NoAttributes, ScalarAttributes, op,
};
use incin_core::exec::{ExecutionContext, TensorHandle};
use incin_core::prelude::{BackendError, ErrorMessage};
use incin_core::shapes::error::OperationKind;
use incin_core::shapes::{Dyn, ShapeBuf};
use incin_core::tensor::base::Tensor;
use incin_core::tensor::device::{Cpu, Device};
use incin_core::tensor::dtype::{DType, FloatDType};
use incin_core::tensor::grad::{Grad, RequiresGrad};
use incin_macros::s;

/// `y = s * x^2`, where `s` is the descriptor's scale. The dtype rides in a
/// `PhantomData` parameter rather than only in the associated type, because an
/// impl type parameter that appears nowhere in the self type is rejected
/// (E0207): `type Dtype = K` alone does not constrain `K`.
///
/// The scale lives in [`Operation::Attributes`] rather than in a constant so
/// that `backward` has to read it from the invocation. `d(s x^2)/dx = 2 s x`
/// depends on the configuration, which is the shape of operation that could
/// not be written before `backward` received the attributes.
#[derive(Debug, Clone)]
struct ScaledSquare<K>(PhantomData<K>);

impl<K: DType> Operation for ScaledSquare<K> {
    type Attributes = ScalarAttributes;

    const KEY: OperationKey = OperationKey {
        namespace: std::borrow::Cow::Borrowed("company.example"),
        name: std::borrow::Cow::Borrowed("scaled_square_any"),
        version: 1,
    };

    fn infer_outputs(
        _attributes: &Self::Attributes,
        inputs: &[LogicalTensorMeta],
    ) -> core::result::Result<Vec<LogicalTensorMeta>, DescriptorError> {
        Ok(inputs.first().cloned().into_iter().collect())
    }
}

/// Dispatch one built-in operation over already-owned storage of dtype `K`.
fn run<O, B, K>(
    attributes: O::Attributes,
    inputs: &[&B::Storage<K>],
) -> core::result::Result<B::Storage<K>, BackendError>
where
    O: Operation,
    B: Backend + ExecuteInto<O, K> + SupportsDType<K>,
    K: DType,
    B::Storage<K>: 'static,
{
    let ctx = ExecutionContext::from_scope(B::default());
    let handles: Vec<TensorHandle<'_>> = inputs
        .iter()
        .map(|s| TensorHandle::from_storage::<B, K, incin_core::dist::Local>(s))
        .collect();
    B::dispatch_into(&ctx, attributes, &handles).map_err(|error| BackendError::Execution {
        operation: OperationKind::Pointwise,
        message: ErrorMessage::new(error.to_string()),
    })
}

/// One impl, one recipe. `B` is any backend that can multiply, `K` any float
/// dtype it supports.
impl<B, K> DifferentiableOp<B> for ScaledSquare<K>
where
    // Two bounds, not four. `ExecuteInto<O, K>` says both that the backend
    // executes the operation and that its result converts to storage, and it
    // carries `Capabilities` as a supertrait, so dispatch asks for nothing
    // more. A fifth built-in inside the recipe would add one line here rather
    // than two.
    B: Backend
        + SupportsDType<K>
        + RecordingBackend<K>
        + ExecuteInto<op::Mul, K>
        + ExecuteInto<op::MulScalar, K>,
    K: FloatDType,
    B::Storage<K>: core::any::Any + StorageOutput + TapeStorage + Send + Sync,
{
    type Dtype = K;
    type Saved = B::Storage<K>;

    fn forward(
        inputs: &[B::Storage<K>],
        attributes: &ScalarAttributes,
    ) -> core::result::Result<(B::Storage<K>, Self::Saved), BackendError> {
        let x = inputs.first().ok_or(BackendError::InvalidInput {
            operation: OperationKind::Pointwise,
            reason: "scaled_square requires one input",
        })?;
        let squared = run::<op::Mul, B, K>(NoAttributes, &[x, x])?;
        let scaled = run::<op::MulScalar, B, K>(
            ScalarAttributes {
                value: attributes.value,
            },
            &[&squared],
        )?;
        // `Saved` holds what the kernel computed with. The scale is not in it:
        // that is the descriptor's, and `backward` is handed the descriptor.
        Ok((scaled, x.clone()))
    }

    fn backward(
        saved: &Self::Saved,
        attributes: &ScalarAttributes,
        grad_out: &B::Storage<K>,
    ) -> incin_core::error::Result<Vec<B::Storage<K>>> {
        let slope = run::<op::MulScalar, B, K>(
            ScalarAttributes {
                value: 2.0 * attributes.value,
            },
            &[saved],
        )
        .map_err(incin_core::error::Error::Backend)?;
        let grad = run::<op::Mul, B, K>(NoAttributes, &[&slope, grad_out])
            .map_err(incin_core::error::Error::Backend)?;
        Ok(vec![grad])
    }
}

type B = CpuBackendImpl<Cpu>;

fn leaf<K: DType<Arg = ()>>(storage: CpuStorage) -> Tensor<Dyn, B, K, Grad> {
    Tensor::<Dyn, B, K, Grad>::try_from_storage(
        storage,
        ShapeBuf::from_slice(&[4]),
        <K as DType>::init(()),
        <Cpu as Device>::init(()),
        <Grad as RequiresGrad>::init(()),
    )
    .expect("the leaf builds")
}

#[test]
fn one_impl_reaches_the_tensor_api() {
    let x = leaf::<f32>(
        CpuStorage::try_from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]), vec![4]).unwrap(),
    );

    // The whole call: one method, no handles, no context, no storage plumbing.
    let y = x
        .apply_op::<ScaledSquare<f32>>(ScalarAttributes { value: 3.0 })
        .expect("apply_op");
    let loss = y.sum_all().expect("reduce to a scalar");
    let grads = loss.backward().expect("backward");
    let gx = grads.require(&x).expect("gradient for x");

    let observed: Vec<f32> = gx.to_vec1().expect("read gradient");
    assert_eq!(observed, vec![6.0f32, 12.0, 18.0, 24.0], "d(3x^2)/dx = 6x");
}

/// The composed kernel's gradient is counted once, not once per built-in
/// operation inside it.
///
/// `forward` here dispatches two built-in differentiable operations, so
/// without a disabled recording scope around it the tape holds their nodes as
/// well as the custom node, all three keyed to the same output. The reverse
/// walk then invokes every one of them against the same output gradient. The
/// observable result is not a crash or a missing gradient but a plausible
/// number that is exactly twice the right one, which no shape or arity check
/// can see.
#[test]
fn a_composed_kernel_does_not_count_its_gradient_twice() {
    use incin_core::backend_authoring::AutogradBackend;

    let x =
        CpuStorage::try_from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]), vec![4]).unwrap();
    let x_id = x.id();
    let ctx = ExecutionContext::new(CpuBackendImpl::<Cpu>::new());
    let handle = TensorHandle::from_storage::<B, f32, incin_core::dist::Local>(&x);

    let y = incin_core::backend_authoring::execute::<ScaledSquare<f32>, B>(
        &ctx,
        ScalarAttributes { value: 3.0 },
        &[handle],
    )
    .expect("the composed kernel runs");

    let grads = <B as AutogradBackend>::backward::<f32>(&y).expect("backward");
    let gx = grads.get(x_id).expect("gradient for x");
    let observed: Vec<f64> = (0..4).map(|i| gx.get(&[i])).collect();
    assert_eq!(
        observed,
        vec![6.0, 12.0, 18.0, 24.0],
        "6x, not 12x: the two built-ins inside `forward` must record nothing"
    );
}

#[test]
fn the_same_impl_covers_a_second_dtype() {
    // Below the tensor API rather than through it: the CPU `reduction` group
    // is declared f32-only in `capability/declarations.rs`, so `sum_all` on an
    // f64 operand is refused by design. That is a built-in coverage boundary,
    // not a property of the custom operation, so the seed is applied directly.
    use incin_core::backend_authoring::AutogradBackend;

    let x =
        CpuStorage::try_from_contiguous(CpuBuffer::F64(vec![1.0, 2.0, 3.0, 4.0]), vec![4]).unwrap();
    let x_id = x.id();
    let ctx = ExecutionContext::new(CpuBackendImpl::<Cpu>::new());
    let handle = TensorHandle::from_storage::<B, f64, incin_core::dist::Local>(&x);

    let y = incin_core::backend_authoring::execute::<ScaledSquare<f64>, B>(
        &ctx,
        ScalarAttributes { value: 3.0 },
        &[handle],
    )
    .expect("the same impl runs on f64");

    let forward: Vec<f64> = (0..4).map(|i| y.get(&[i])).collect();
    assert_eq!(forward, vec![3.0, 12.0, 27.0, 48.0], "3x^2 in f64");

    let grads = <B as AutogradBackend>::backward::<f64>(&y).expect("backward");
    let gx = grads.get(x_id).expect("gradient for x");
    let observed: Vec<f64> = (0..4).map(|i| gx.get(&[i])).collect();
    assert_eq!(
        observed,
        vec![6.0, 12.0, 18.0, 24.0],
        "the same recipe, in f64"
    );
}

/// The recorded recipe reads the scale from the invocation it belongs to.
///
/// Two calls to the same operation with different descriptors, both live on
/// the tape at once. `d(s x^2)/dx = 2 s x`, so each node's gradient is a
/// different multiple of its input, and a `backward` that read the scale from
/// anywhere other than its own attributes would give both nodes the same one.
///
/// This is what the attributes parameter buys. Before it, the only way to
/// reach the scale from `backward` was to copy it into `Saved`, which puts the
/// configuration on the node twice and makes `Saved` mean both what the kernel
/// computed and what the caller asked for.
#[test]
fn each_recorded_node_reads_its_own_attributes() {
    use incin_core::backend_authoring::AutogradBackend;

    let ctx = ExecutionContext::new(CpuBackendImpl::<Cpu>::new());

    let run_with = |scale: f64| {
        let x = CpuStorage::try_from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]), vec![4])
            .unwrap();
        let x_id = x.id();
        let handle = TensorHandle::from_storage::<B, f32, incin_core::dist::Local>(&x);
        let y = incin_core::backend_authoring::execute::<ScaledSquare<f32>, B>(
            &ctx,
            ScalarAttributes { value: scale },
            &[handle],
        )
        .expect("the scaled kernel runs");
        (x_id, y)
    };

    // Both nodes are recorded before either is walked, so neither backward can
    // be reading an ambient "most recent" configuration.
    let (x3_id, y3) = run_with(3.0);
    let (x5_id, y5) = run_with(5.0);

    let g3 = <B as AutogradBackend>::backward::<f32>(&y3).expect("backward at scale 3");
    let observed3: Vec<f64> = (0..4)
        .map(|i| g3.get(x3_id).expect("gradient for x3").get(&[i]))
        .collect();
    assert_eq!(observed3, vec![6.0, 12.0, 18.0, 24.0], "2 * 3 * x");

    let g5 = <B as AutogradBackend>::backward::<f32>(&y5).expect("backward at scale 5");
    let observed5: Vec<f64> = (0..4)
        .map(|i| g5.get(x5_id).expect("gradient for x5").get(&[i]))
        .collect();
    assert_eq!(observed5, vec![10.0, 20.0, 30.0, 40.0], "2 * 5 * x");
}

/// `y = [a..., b...]` joined along axis 0: two inputs, and an output longer
/// than either of them.
///
/// The shape of operation `apply_op` cannot carry. Its two operands have
/// different geometry and its output has a third, so none of the three can be
/// inferred from a receiver. Written the same way as [`ScaledSquare`], out of
/// dispatched built-ins, so the recipe is backend-generic: `concat` forward,
/// two `narrow`s of the output gradient backward.
#[derive(Debug, Clone)]
struct Concat2<K>(PhantomData<K>);

impl<K: DType> Operation for Concat2<K> {
    type Attributes = NoAttributes;

    const KEY: OperationKey = OperationKey {
        namespace: std::borrow::Cow::Borrowed("company.example"),
        name: std::borrow::Cow::Borrowed("concat2_any"),
        version: 1,
    };

    fn infer_outputs(
        _attributes: &Self::Attributes,
        inputs: &[LogicalTensorMeta],
    ) -> core::result::Result<Vec<LogicalTensorMeta>, DescriptorError> {
        let joined: usize = inputs.iter().map(leading_extent).sum();
        Ok(vec![LogicalTensorMeta {
            shape: Some(ShapeBuf::from_slice(&[joined])),
            ..inputs
                .first()
                .cloned()
                .unwrap_or_else(LogicalTensorMeta::unknown)
        }])
    }
}

/// Length of a rank-1 operand, or zero when the caller pinned no shape.
fn leading_extent(meta: &LogicalTensorMeta) -> usize {
    meta.shape
        .as_ref()
        .and_then(|shape| shape.as_ref().first().copied())
        .unwrap_or(0)
}

impl<B, K> DifferentiableOp<B> for Concat2<K>
where
    B: Backend
        + SupportsDType<K>
        + RecordingBackend<K>
        + ExecuteInto<op::ConcatExact, K>
        + ExecuteInto<op::Narrow, K>,
    K: FloatDType,
    B::Storage<K>: core::any::Any + StorageOutput + TapeStorage + Send + Sync,
{
    type Dtype = K;
    /// The two operand lengths, which is what the split in `backward` needs
    /// and all it needs. Not the operands themselves: a concatenation's
    /// gradient does not depend on the values it joined.
    type Saved = (usize, usize);

    fn forward(
        inputs: &[B::Storage<K>],
        _attributes: &NoAttributes,
    ) -> core::result::Result<(B::Storage<K>, Self::Saved), BackendError> {
        let [left, right] = inputs else {
            return Err(BackendError::InvalidInput {
                operation: OperationKind::Concat,
                reason: "concat2 requires exactly two inputs",
            });
        };
        let lengths = (extent_of::<B, K>(left), extent_of::<B, K>(right));
        let joined = run::<op::ConcatExact, B, K>(AxisAttributes { axis: 0 }, &[left, right])?;
        Ok((joined, lengths))
    }

    fn backward(
        saved: &Self::Saved,
        _attributes: &NoAttributes,
        grad_out: &B::Storage<K>,
    ) -> incin_core::error::Result<Vec<B::Storage<K>>> {
        let (left, right) = *saved;
        // One narrow per operand, each over the span that operand contributed.
        // In operand order, because the reverse walk pairs the returned
        // gradients with the recorded inputs positionally.
        let grad_left = run::<op::Narrow, B, K>(
            NarrowAttributes {
                axis: 0,
                start: 0,
                length: left,
            },
            &[grad_out],
        )
        .map_err(incin_core::error::Error::Backend)?;
        let grad_right = run::<op::Narrow, B, K>(
            NarrowAttributes {
                axis: 0,
                start: left,
                length: right,
            },
            &[grad_out],
        )
        .map_err(incin_core::error::Error::Backend)?;
        Ok(vec![grad_left, grad_right])
    }
}

/// Leading extent of a rank-1 storage buffer.
fn extent_of<B: Backend, K: DType>(storage: &B::Storage<K>) -> usize {
    B::shape(storage).as_ref().first().copied().unwrap_or(0)
}

fn static_leaf<S: incin_core::shapes::Shape>(values: Vec<f32>) -> Tensor<S, B, f32, Grad> {
    let dims = vec![values.len()];
    Tensor::<S, B, f32, Grad>::try_from_storage(
        CpuStorage::try_from_contiguous(CpuBuffer::F32(values), dims.clone()).expect("storage"),
        ShapeBuf::from_slice(&dims),
        <f32 as DType>::init(()),
        <Cpu as Device>::init(()),
        <Grad as RequiresGrad>::init(()),
    )
    .expect("the leaf builds")
}

/// Two operands of different shapes and a third shape out, in one call.
///
/// The operands are `s![4]` and `s![2]` and the result is `s![6]`, so all
/// three shapes are distinct Rust types. That is what rules out a slice of
/// `&Self` for the extra operands: it would make every operand share the
/// receiver's shape and layout, and this call could not be written at all.
/// Dispatch never asked for that uniformity, since a handle is built from the
/// backend, the dtype and the placement alone.
///
/// The gradient is read through a second custom operation rather than a bare
/// `sum_all`, so the seed reaching `Concat2::backward` varies along the joined
/// axis. `d(z^2)/dz = 2z`, so each operand's gradient is twice its own values
/// and nothing else's: swapping the two narrows, or sliding either range,
/// moves numbers that a uniform seed would leave identical.
#[test]
fn a_two_input_shape_changing_operation_is_one_call() {
    use incin_core::shapes::ShapeValue;

    let x = static_leaf::<s![4]>(vec![1.0, 2.0, 3.0, 4.0]);
    let w = static_leaf::<s![2]>(vec![10.0, 20.0]);

    // No handle, no execution context, and the dtype, device and gradient
    // marker are not restated. Only the output shape is, because only the
    // caller knows it.
    let joined = x
        .apply_op_n::<Concat2<f32>, s![6]>(
            &[w.inner()],
            NoAttributes,
            ShapeValue::try_new(ShapeBuf::from_slice(&[6])).expect("the output shape"),
        )
        .expect("apply_op_n");

    let forward: Vec<f32> = joined.to_vec1().expect("read the join");
    assert_eq!(forward, vec![1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0]);

    let squared = joined
        .apply_op::<ScaledSquare<f32>>(ScalarAttributes { value: 1.0 })
        .expect("apply_op");
    let loss = squared.sum_all().expect("reduce to a scalar");
    let grads = loss.backward().expect("backward");

    let gx: Vec<f32> = grads
        .require(&x)
        .expect("gradient for x")
        .to_vec1()
        .unwrap();
    let gw: Vec<f32> = grads
        .require(&w)
        .expect("gradient for w")
        .to_vec1()
        .unwrap();
    assert_eq!(gx, vec![2.0f32, 4.0, 6.0, 8.0], "2x over the first span");
    assert_eq!(gw, vec![20.0f32, 40.0], "2w over the second span");
}
