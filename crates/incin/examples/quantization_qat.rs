//! The #93 quantization boundary, end to end at the tensor level: Q8_0
//! blocks, `quantize`/`dequantize`, `quantized_matmul`, and the
//! straight-through estimator (STE) gradient that makes a quantized
//! training pass expressible.
//!
//! Everything here follows from one packing decision: a `Q8_0` block stores
//! 32 consecutive `f32` values as 34 bytes - one `f16` scale and 32 `i8`
//! quants (`StorageEncoding::block(32, 34, 2)`). The block axis must be the
//! *last* axis and a whole multiple of 32; the round trip is approximate
//! because the scale is `f16`; and backward cannot differentiate rounding,
//! so it passes the cotangent through unchanged - PyTorch QAT's
//! `FakeQuantize` rule, spelled `GradientRule::StraightThrough` in the
//! operation catalog. Section 4 also prints the one honest limitation: the
//! tensor facade cannot yet hold a `Grad`-marked `Q8_0` tensor, so the
//! straight-through walk is exercised on the backend dispatch surface
//! behind it, exactly as the contract tests do.
//!
//! Run with: `cargo run -p incin --example quantization_qat --no-default-features --features incin-backends/cpu,incin/cpu`

#![cfg(feature = "cpu")]

use incin::prelude::*;
use incin_backends::cpu::{CpuBuffer, CpuStorage, tape_depth};
use incin_core::backend_authoring::AutogradBackend;
use incin_core::dist::Local;
use incin_core::exec::catalog::{NoAttributes, QuantizationAttributes, op};
use incin_core::exec::{ExecutionContext, GradMode, TensorHandle, dispatch};

type B = DefaultBackend;

fn main() -> incin::Result<()> {
    section("1. A float workload, then the same values as Q8_0 blocks");
    blocks()?;

    section("2. The round trip: decode, inspect, measure the error");
    round_trip()?;

    section("3. Matrix products on quantized operands");
    quantized_matmul()?;

    section("4. The straight-through estimator: backward through the boundary");
    ste_gradient()?;

    Ok(())
}

/// Deterministic weights, so every printed number in this example is
/// reproducible. Values stay in [-1, 1], the range a per-block
/// `max_abs / 127` scale represents well.
fn weight_values() -> Vec<f32> {
    (0..32 * 32)
        .map(|index| ((index % 23) as f32 - 11.0) / 11.0)
        .collect()
}

fn activation_values() -> Vec<f32> {
    (0..2 * 32)
        .map(|index| (index as f32 - 32.0) / 32.0)
        .collect()
}

/// Section 1: `quantize` is a dtype conversion with a layout rule, not a
/// different tensor type. The shape does not change, the dtype does, and
/// the storage collapses from four bytes per value to 34 bytes per 32.
fn blocks() -> incin::Result<()> {
    // A one-layer linear workload: activations `x [2, 32]` times weights
    // `w [32, 32]`. The extent 32 is exactly one Q8_0 block, so every row
    // of `w` is a whole number of blocks along the last axis - which is
    // the rule `quantize` enforces at the call site.
    let x = Tensor::<s![2, 32], B>::from_slice(&activation_values(), ())?;
    let w = Tensor::<s![32, 32], B>::from_slice(&weight_values(), ())?;
    let reference = x.matmul(&w)?;

    println!("  x:    shape {:?}, dtype {}", x.dims(), x.dtype().name());
    println!("  w:    shape {:?}, dtype {}", w.dims(), w.dtype().name());
    println!(
        "  x @ w: shape {:?} (plain f32 matmul, the reference)",
        reference.dims()
    );

    // Compress `w` along its last axis. `-1` is the only axis Q8_0 blocks
    // can run along: the kernel reads the flat row-major buffer, and a last
    // axis divisible by 32 is exactly the condition under which 32-element
    // chunks never straddle two rows.
    let qw = w.quantize(-1)?;
    println!(
        "\n  w.quantize(-1) -> shape {:?}, dtype {}",
        qw.dims(),
        qw.dtype().name()
    );

    let encoding = qw.dtype().encoding();
    let elements = 32 * 32;
    let block_count = elements / encoding.logical_elements_per_block();
    let float_bytes = elements * core::mem::size_of::<f32>();
    let quant_bytes = block_count * encoding.bytes_per_block();
    println!(
        "  storage: {elements} f32 values = {float_bytes} bytes; \
         Q8_0: {block_count} blocks x {} bytes = {quant_bytes} bytes ({:.1}x smaller)",
        encoding.bytes_per_block(),
        float_bytes as f64 / quant_bytes as f64,
    );
    println!(
        "  one block = 32 values packed as one f16 scale + 32 i8 quants; \
         there is no per-element f32 left to read without decoding"
    );

    // A block tensor prints its own refusal: values are not addressable
    // until `dequantize` expands them again.
    println!("  printing it: {qw}");

    Ok(())
}

/// Section 2: `dequantize` is the exact inverse operation, but not an
/// exact inverse *value*: the block scale was rounded to `f16` on the way
/// in, so the round trip is approximate by construction.
fn round_trip() -> incin::Result<()> {
    let w = Tensor::<s![32, 32], B>::from_slice(&weight_values(), ())?;
    let restored = w.quantize(-1)?.dequantize::<f32>()?;

    println!(
        "  dequantize -> shape {:?}, dtype {}",
        restored.dims(),
        restored.dtype().name()
    );

    let original = w.to_vec1::<f32>()?;
    let decoded = restored.to_vec1::<f32>()?;
    let mut max_error = 0.0f32;
    for (index, (before, after)) in original.iter().zip(&decoded).enumerate() {
        max_error = max_error.max((before - after).abs());
        if index < 4 {
            println!("  w[0][{index}] = {before:>8.5}  ->  {after:>8.5}");
        }
    }
    println!(
        "  max |original - decoded| over all {} values = {max_error:.6}",
        original.len()
    );
    println!("  the f16 block scale rounds, so decoding is approximate - never bit-exact");

    Ok(())
}

/// Section 3: two facts a learner should see side by side. `Tensor::matmul`
/// admits one dtype for both operands and dispatches the *float* MatMul
/// descriptor, so Q8_0 operands are refused by the catalog's float rule.
/// The operation built for blocks, `quantized_matmul`, is real and runs -
/// but it is only reachable through the backend-authoring dispatch, not
/// through a `Tensor` method yet, and it is forward-only.
fn quantized_matmul() -> incin::Result<()> {
    let x = Tensor::<s![2, 32], B>::from_slice(&activation_values(), ())?;
    let w = Tensor::<s![32, 32], B>::from_slice(&weight_values(), ())?;
    let reference = x.matmul(&w)?;

    let qx = x.quantize(-1)?;
    let qw = w.quantize(-1)?;

    // (a) The refusal the dtype bounds do NOT catch: both operands are
    // Q8_0, so the shapes and the shared `K` line up and the call compiles.
    // It is the operation catalog that refuses at run time, naming the
    // rule rather than silently decoding behind your back.
    match qw.matmul(&qw) {
        Ok(_) => println!("  qw.matmul(qw) succeeded, which the float rule does not claim"),
        Err(error) => println!("  Tensor::matmul on Q8_0 operands refused: {error}"),
    }

    // (b) The real path: `quantized_matmul` consumes two Q8_0 tensors and
    // writes f32. There is no `Tensor::quantized_matmul` method in this
    // release - the descriptor is reached through the same
    // backend-authoring dispatch the capability tables and tests speak,
    // which is worth knowing before you go looking for a facade method.
    let context = ExecutionContext::<B>::new(B::default());
    let lhs = TensorHandle::from_storage::<B, Q8_0, Local>(qx.inner());
    let rhs = TensorHandle::from_storage::<B, Q8_0, Local>(qw.inner());
    let product_storage =
        dispatch::execute::<op::QuantizedMatMul, B>(&context, NoAttributes, &[lhs, rhs])?;
    let product = Tensor::<s![2, 32], B>::from_raw(product_storage, ())?;

    println!(
        "  quantized_matmul: Q8_0 [2,32] x Q8_0 [32,32] -> shape {:?}, dtype {}",
        product.dims(),
        product.dtype().name()
    );
    let reference_values = reference.to_vec1::<f32>()?;
    let product_values = product.to_vec1::<f32>()?;
    let max_error = reference_values
        .iter()
        .zip(&product_values)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "  max |f32 matmul - quantized matmul| = {max_error:.6} \
         (the same block rounding you saw in section 2, now inside the product)"
    );

    // Fail-closed: `quantized_matmul` has no gradient rule at all
    // (`GradientRule::None`), so its capability row keeps `training = false`
    // and a training-mode invocation is refused at admission rather than
    // admitted into a graph with a hole where its backward should be.
    let training = ExecutionContext::<B>::new(B::default()).with_training(true);
    let train_lhs = TensorHandle::from_storage::<B, Q8_0, Local>(qx.inner());
    let train_rhs = TensorHandle::from_storage::<B, Q8_0, Local>(qw.inner());
    match dispatch::execute::<op::QuantizedMatMul, B>(
        &training,
        NoAttributes,
        &[train_lhs, train_rhs],
    ) {
        Ok(_) => println!("  training mode accepted quantized_matmul, which the contract forbids"),
        Err(error) => println!("  training mode refuses quantized_matmul: {error}"),
    }

    Ok(())
}

/// Section 4: the gradient half of #93, and the honest line between the
/// facade and the backend.
///
/// The rule: forward computes the true block encoding and its decode,
/// backward passes the cotangent through *unchanged* (`grad_in = grad_out`)
/// because rounding has no derivative and Q8_0's per-block scale never
/// saturates - there is no clip range to zero outside of. That is PyTorch
/// QAT's `FakeQuantize` rule, and the CPU backend records it: one straight-
/// through tape node per half of the boundary, so `dequantize(quantize(x))`
/// backward is exactly the identity.
///
/// The line: a `Tensor` may only carry gradient tracking for a *float*
/// dtype, so a `Grad`-marked tensor quantizing into `Q8_0` is refused at
/// construction (printed below), while a `NoGrad` operand records nothing.
/// The STE walk is therefore exercised through the same backend dispatch
/// the `crates/incin-backends/tests/quantize_ste.rs` contract tests use -
/// that is where the gradient rule lives until the facade can hold a
/// quantized tensor under gradient tracking.
fn ste_gradient() -> incin::Result<()> {
    let values: Vec<f32> = (0..64).map(|index| (index as f32 - 32.0) / 32.0).collect();
    let cotangent: Vec<f32> = (0..64).map(|index| 1.0 + index as f32 / 64.0).collect();

    // (a) The facade attempt. Gradient tracking on a quantized dtype is
    // refused with the dtype named: the Q8_0 result would inherit this
    // tensor's Grad marker, and `validate_gradient_dtype` only admits
    // float dtypes to it.
    let x = Tensor::<s![64], B>::from_slice(&values, ())?.require_grad();
    match x.quantize(-1) {
        Ok(_) => println!("  x.quantize(-1) on a Grad tensor was accepted"),
        Err(error) => println!("  x.quantize(-1) on a Grad tensor refused: {error}"),
    }

    // (b) The rule itself, where it lives. `GradMode::Enabled` is the
    // ambient default; the explicit scope spells it out rather than
    // depending on it. Quantize and dequantize each record one tape node.
    let context = ExecutionContext::<B>::new(B::default());
    let input = CpuStorage::try_from_contiguous(CpuBuffer::F32(values), vec![64])?;
    let weights = CpuStorage::try_from_contiguous(CpuBuffer::F32(cotangent.clone()), vec![64])?;

    let before = tape_depth();
    let restored = GradMode::Enabled.scope(|| -> incin::Result<CpuStorage> {
        let blocks = dispatch::execute::<op::Quantize, B>(
            &context,
            QuantizationAttributes {
                dtype: DTypeId::Q8_0.descriptor(),
            },
            &[TensorHandle::from_storage::<B, f32, Local>(&input)],
        )?;
        dispatch::execute::<op::Dequantize, B>(
            &context,
            QuantizationAttributes {
                dtype: DTypeId::F32.descriptor(),
            },
            &[TensorHandle::from_storage::<B, Q8_0, Local>(&blocks)],
        )
        .map_err(Into::into)
    })?;
    println!(
        "  tape nodes recorded by quantize + dequantize: {}",
        tape_depth() - before
    );

    // Seed the loss outside the boundary scope: sum(restored * weights),
    // so the cotangent arriving at the boundary is exactly `weights`.
    let weighted = dispatch::execute::<op::Mul, B>(
        &context,
        NoAttributes,
        &[
            TensorHandle::from_storage::<B, f32, Local>(&restored),
            TensorHandle::from_storage::<B, f32, Local>(&weights),
        ],
    )?;
    let loss = dispatch::execute::<op::SumAll, B>(
        &context,
        NoAttributes,
        &[TensorHandle::from_storage::<B, f32, Local>(&weighted)],
    )?;
    let grads = B::backward::<f32>(&loss)?;
    let grad = B::get_grad::<f32>(&input, &grads)?
        .expect("the quantized operand's input received a gradient");
    for index in [0usize, 16, 32, 48] {
        println!(
            "  grad[{index}] = {:>8.5}   incoming cotangent = {:>8.5}",
            grad.get(&[index]),
            cotangent[index]
        );
    }
    for (index, cotangent_value) in cotangent.iter().enumerate() {
        assert_eq!(
            grad.get(&[index]),
            f64::from(*cotangent_value),
            "STE must pass the cotangent through unchanged at element {index}"
        );
    }
    println!(
        "  all {} gradients bit-identical to the incoming cotangent: \
         backward through quantize/dequantize is the identity",
        cotangent.len()
    );
    println!(
        "  an approximation, and the catalog labels it as one: \
         GradientRule::StraightThrough, rendered as STE in OPERATION_SEMANTICS"
    );

    Ok(())
}

fn section(title: &str) {
    println!("\n{title}");
    println!("{}", "-".repeat(title.len()));
}
