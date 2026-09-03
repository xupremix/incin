//! Practical Example: Dynamic-to-Static Bridging & DType Conversions.
//!
//! In production systems, requests arrive with dynamic runtime dimensions (variable
//! batch sizes, token lengths from HTTP requests) and dynamic dtypes (quantized / float).
//! However, internal neural network kernels perform best and guarantee correctness
//! when bound to compile-time static shape proofs.
//!
//! This example demonstrates:
//! 1. Ingesting variable dynamic-shaped input (`Tensor<Dyn, Backend>`)
//! 2. Reshaping with inference using `reshape_infer(shape![B, infer])`
//! 3. Re-asserting static compile-time type proofs with `to_shape::<s![B, 784]>()`
//! 4. Handling dimension mismatch errors safely without panicking
//! 5. Converting between dtypes (`f32` -> `f64` -> `i64`) via `to_dtype::<K>()`
//! 6. Running a compile-time statically-typed MLP from a dynamic input boundary.
//!
//! Run with: `cargo run -p incin --example dynamic_to_static_bridging --features cpu`

#![allow(missing_docs)]
#![allow(clippy::type_complexity)]

use incin::nn::Linear;
use incin::prelude::*;

type Backend = DefaultBackend;

/// A statically-typed perception layer expecting exactly 784 inputs (e.g. 28x28 image)
/// and producing 10 features with compile-time shape proofs.
#[module]
struct StaticClassifier {
    fc1: Linear<s![784, 128], Backend>,
    fc2: Linear<s![128, 10], Backend>,
}

impl StaticClassifier {
    pub fn new() -> incin::Result<Self> {
        Ok(Self {
            fc1: Linear::build(())?,
            fc2: Linear::build(())?,
        })
    }

    /// Forward pass strictly verified at compile-time: inputs must be `[4, 784]`.
    pub fn forward(
        &self,
        x: Tensor<s![4, 784], Backend>,
    ) -> incin::Result<Tensor<s![4, 10], Backend, f32, Grad>> {
        let h = self.fc1.forward(x)?;
        let a = h.relu()?;
        self.fc2.forward(a)
    }
}

fn main() -> incin::Result<()> {
    println!("=== Practical Example: Dynamic-to-Static Bridging & DType Transitions ===\n");

    // ─────────────────────────────────────────────────────────────────────────
    // 1. Dynamic Ingestion & Shape Inference
    // ─────────────────────────────────────────────────────────────────────────
    println!("[1] Ingesting dynamic runtime batch (e.g. incoming JSON payload)...");
    let runtime_batch_size = 4;
    let raw_flat_data = vec![0.5_f32; runtime_batch_size * 28 * 28];

    // Allocate dynamic tensor from host vector
    let dynamic_images: Tensor<Dyn, Backend> =
        Cpu.tensor_from_vec(raw_flat_data, [runtime_batch_size, 1, 28, 28])?;
    println!(
        "  • Ingested dynamic tensor: dims = {:?}",
        dynamic_images.dims()
    );

    // Flatten spatial dimensions dynamically: [4, 1, 28, 28] -> [4, 784]
    let flattened = dynamic_images.reshape([runtime_batch_size, 784])?;
    println!(
        "  • Flattened dynamic tensor: dims = {:?}",
        flattened.dims()
    );

    // ─────────────────────────────────────────────────────────────────────────
    // 2. Bridging from Dynamic to Static Type Proofs (`to_shape`)
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n[2] Promoting dynamic tensor to compile-time static proof s![4, 784]...");

    // `.to_shape()` verifies that runtime dimensions match the compile-time type,
    // returning a strongly-typed `Tensor<s![4, 784], Backend>`:
    let static_tensor: Tensor<s![4, 784], Backend> = flattened.to_shape::<s![4, 784]>()?;
    println!("  • Type proof acquired! Tensor shape at compile-time is s![4, 784]");

    // Execute the statically-checked model:
    let model = StaticClassifier::new()?;
    let output = model.forward(static_tensor)?;
    println!(
        "  • Statically-typed model output shape: {:?}",
        output.dims()
    );

    // ─────────────────────────────────────────────────────────────────────────
    // 3. Fallible Dimension Safety (Preventing Runtime Shape Invariant Violations)
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n[3] Handling shape mismatch defensively at the dynamic boundary...");
    let wrong_batch: Tensor<Dyn, Backend> = Cpu.zeros([3, 784])?; // Expected 4, got 3

    match wrong_batch.to_shape::<s![4, 784]>() {
        Ok(_) => println!("  • Unexpected success!"),
        Err(err) => println!("  • Correctly caught shape mismatch error at boundary: {err}"),
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 4. DType Conversions (`to_dtype`)
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n[4] Converting and transitioning tensor element dtypes...");
    let f32_tensor = Cpu.tensor([1.5_f32, 2.5, 3.5])?;
    println!(
        "  • Original float32 tensor dtype: {:?}",
        f32_tensor.dtype()
    );

    // Convert f32 -> f64
    let f64_tensor = f32_tensor.to_dtype::<f64>()?;
    println!("  • Converted to f64: {:?}", f64_tensor.to_vec1::<f64>()?);

    // Convert f64 -> i64
    let i64_tensor = f64_tensor.to_dtype::<i64>()?;
    println!("  • Converted to i64: {:?}", i64_tensor.to_vec1::<i64>()?);

    println!("\n[5] Dynamic-to-static bridging & dtype transitions completed successfully!");
    Ok(())
}
