//! The straight-through estimator (STE) at the quantize boundary (#93).
//!
//! Decision 2 of `docs/plan/research/0.2.0/93-quantized-contract.md`: the
//! gradient through `quantize` is PyTorch QAT's rule, not a silent NoGrad.
//! Forward produces the true Q8_0 block encoding; backward passes the
//! cotangent through unchanged (`grad_in = grad_out`) because the rounding
//! has no derivative and Q8_0's per-block scale never saturates, so there is
//! no clip range to zero outside of. `dequantize` carries the dual identity
//! recipe, which makes `dequantize(quantize(x))` backward exactly the
//! identity on in-range gradients.

#![cfg(feature = "cpu")]

use incin_backends::cpu::{CpuBackendImpl, CpuBuffer, CpuStorage, tape_depth};
use incin_core::backend_authoring::AutogradBackend;
use incin_core::exec::catalog::{NoAttributes, QuantizationAttributes, op};
use incin_core::exec::{ExecutionContext, GradMode, TensorHandle, dispatch};
use incin_core::tensor::dtype::DTypeId;

type B = CpuBackendImpl;

fn storage(values: Vec<f32>, shape: Vec<usize>) -> CpuStorage {
    CpuStorage::try_from_contiguous(CpuBuffer::F32(values), shape).expect("well-formed f32 storage")
}

fn handle(storage: &CpuStorage) -> TensorHandle<'_> {
    TensorHandle::from_storage::<B, f32, _>(storage)
}

fn context() -> ExecutionContext<B> {
    ExecutionContext::new(B::new())
}

fn quantize(context: &ExecutionContext<B>, input: &CpuStorage) -> CpuStorage {
    dispatch::execute::<op::Quantize, _>(
        context,
        QuantizationAttributes {
            dtype: DTypeId::Q8_0.descriptor(),
        },
        &[handle(input)],
    )
    .expect("quantize executes")
}

fn dequantize(context: &ExecutionContext<B>, blocks: &CpuStorage) -> CpuStorage {
    dispatch::execute::<op::Dequantize, _>(
        context,
        QuantizationAttributes {
            dtype: DTypeId::F32.descriptor(),
        },
        &[handle(blocks)],
    )
    .expect("dequantize executes")
}

fn sum_all(context: &ExecutionContext<B>, input: &CpuStorage) -> CpuStorage {
    dispatch::execute::<op::SumAll, _>(context, NoAttributes, &[handle(input)])
        .expect("sum_all executes")
}

fn input_values() -> Vec<f32> {
    (0..32).map(|index| index as f32 - 8.0).collect()
}

#[test]
fn quantize_records_one_ste_node_when_grad_mode_records_and_none_when_it_does_not() {
    let context = context();
    let input = storage(input_values(), vec![32]);

    let before = tape_depth();
    GradMode::Enabled.scope(|| {
        let _ = quantize(&context, &input);
    });
    assert_eq!(
        tape_depth(),
        before + 1,
        "quantize must record exactly one STE tape node under GradMode::Enabled"
    );

    let before = tape_depth();
    GradMode::Disabled.scope(|| {
        let _ = quantize(&context, &input);
    });
    assert_eq!(
        tape_depth(),
        before,
        "quantize under GradMode::Disabled must record nothing"
    );
}

#[test]
fn quantize_backward_passes_the_cotangent_through_unchanged() {
    let context = context();
    let input = storage(input_values(), vec![32]);

    let blocks = GradMode::Enabled.scope(|| quantize(&context, &input));
    assert_eq!(blocks.dtype, DTypeId::Q8_0.descriptor());

    // Seed the quantized output directly: the cotangent arrives as f32 with
    // the operand's shape, and STE says every element of the input sees it
    // unchanged - all values are in range because Q8_0 never clips.
    let seed = storage(vec![2.0; 32], vec![32]);
    let grads =
        B::backward_with::<f32>(&blocks, &seed).expect("the seeded walk reaches the STE node");
    let grad = B::get_grad::<f32>(&input, &grads)
        .expect("gradient map returned")
        .expect("the quantized operand's input received a gradient");
    for index in 0..32 {
        assert_eq!(
            grad.get(&[index]),
            2.0,
            "STE must pass the cotangent through unchanged at element {index}"
        );
    }
}

#[test]
fn roundtrip_backward_is_exactly_the_identity_on_the_gradient() {
    let context = context();
    let input = storage(input_values(), vec![32]);

    let before = tape_depth();
    let restored = GradMode::Enabled.scope(|| {
        let blocks = quantize(&context, &input);
        dequantize(&context, &blocks)
    });
    assert_eq!(
        tape_depth(),
        before + 2,
        "both halves of the boundary must record their identity recipe"
    );

    let loss = sum_all(&context, &restored);
    let grads = B::backward::<f32>(&loss).expect("roundtrip backward walks");
    let grad = B::get_grad::<f32>(&input, &grads)
        .expect("gradient map returned")
        .expect("the input received a gradient");
    for index in 0..32 {
        assert_eq!(
            grad.get(&[index]),
            1.0,
            "an all-ones seed must arrive at the input unchanged at element {index}"
        );
    }
}

#[test]
fn roundtrip_backward_transports_an_arbitrary_cotangent_exactly() {
    let context = context();
    let input = storage(input_values(), vec![32]);
    // Exactly representable weights, so the assertion is bit equality rather
    // than a tolerance: identity STE composes with no rounding of its own.
    let weights: Vec<f32> = (0..32)
        .map(|index| match index % 4 {
            0 => 0.5,
            1 => 1.5,
            2 => -2.0,
            _ => 3.25,
        })
        .collect();
    let weight_storage = storage(weights.clone(), vec![32]);

    let loss = GradMode::Enabled.scope(|| {
        let blocks = quantize(&context, &input);
        let restored = dequantize(&context, &blocks);
        let weighted = dispatch::execute::<op::Mul, _>(
            &context,
            NoAttributes,
            &[handle(&restored), handle(&weight_storage)],
        )
        .expect("mul executes");
        sum_all(&context, &weighted)
    });

    let grads = B::backward::<f32>(&loss).expect("weighted roundtrip backward walks");
    let grad = B::get_grad::<f32>(&input, &grads)
        .expect("gradient map returned")
        .expect("the input received a gradient");
    for (index, expected) in weights.iter().enumerate() {
        assert_eq!(
            grad.get(&[index]),
            f64::from(*expected),
            "the cotangent must arrive unchanged at element {index}"
        );
    }
}

/// The capability row and the kernel are one claim (#93).
///
/// `dispatch::execute` admits against `context.training()` before any
/// executor runs, so a row whose `training` flag stayed `false` refused
/// `quantize` with "training is unsupported for quantize" even though
/// `cpu/canonical/linalg.rs` was already pushing the straight-through
/// entry. This is that probe, kept as a regression: a training-mode
/// context executes both halves of the boundary and records one STE node
/// per half.
#[test]
fn a_training_context_admits_the_boundary_and_records_both_ste_entries() {
    let context = ExecutionContext::<B>::new(B::new()).with_training(true);
    let input = storage(input_values(), vec![32]);

    let before = tape_depth();
    let blocks = GradMode::Enabled.scope(|| quantize(&context, &input));
    assert_eq!(blocks.dtype, DTypeId::Q8_0.descriptor());
    let restored = GradMode::Enabled.scope(|| dequantize(&context, &blocks));
    assert_eq!(restored.dtype, DTypeId::F32.descriptor());
    assert_eq!(
        tape_depth(),
        before + 2,
        "training mode must record one STE node per half of the boundary"
    );
}

/// The fail-closed half of the same contract: `quantized_matmul` has no
/// gradient rule (`GradientRule::None`) and its CPU kernel records
/// nothing, so its row keeps `training = false` and a training-mode
/// invocation is still refused at admission rather than admitted into a
/// graph with a hole where its backward should be.
#[test]
fn a_training_context_still_refuses_quantized_matmul() {
    let context = ExecutionContext::<B>::new(B::new()).with_training(true);
    // The descriptor validates before any capability query, so the operands
    // have to satisfy `OutputRule::MatMul` (`lhs[-1] == rhs[-2]`) for this
    // to reach the training check rather than fail on shape. Two rows of
    // thirty-two elements are one whole Q8_0 block each.
    let left_row = storage((0..32).map(|index| index as f32).collect(), vec![1, 32]);
    let right_row = storage((0..32).map(|index| index as f32).collect(), vec![32, 1]);
    let lhs = GradMode::Enabled.scope(|| quantize(&context, &left_row));
    let rhs = GradMode::Enabled.scope(|| quantize(&context, &right_row));

    let error = dispatch::execute::<op::QuantizedMatMul, _>(
        &context,
        NoAttributes,
        &[handle(&lhs), handle(&rhs)],
    )
    .expect_err("quantized_matmul records no tape, so training mode must refuse it");
    assert!(
        error.to_string().contains("training is unsupported"),
        "the refusal must name the missing training coverage, got: {error}"
    );
}
