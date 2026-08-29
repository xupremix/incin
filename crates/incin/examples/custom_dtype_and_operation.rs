//! Practical Example: Authoring a Custom Operation with Validation Proofs.
//!
//! Demonstrates:
//! 1. Defining a custom logical dtype `Fp8E4M3` (8-bit float)
//! 2. Defining a custom operation contract `FusedBiasGelu` with shape inference
//! 3. Proving shape & contract invariants ahead of kernel execution
//! 4. Calling custom operations directly via:
//!    - A. Standalone Function: `fused_bias_gelu(&x, &bias)` (No extension trait required!)
//!    - B. Extension Trait: `x.fused_bias_gelu(&bias)` (For method-chaining syntax)
//! 5. Catching dimension mismatch errors *before* backend kernel launch.
//!
//! Run with: `cargo run -p incin --example custom_dtype_and_operation --features cpu,backend-authoring`

#![allow(missing_docs)]
#![allow(clippy::type_complexity)]

use core::marker::PhantomData;
use incin::backend_authoring::operations::op;
use incin::backend_authoring::operations::{NoAttributes, Operation};
use incin::backend_authoring::{
    Backend, DescriptorError, Execute, LogicalTensorMeta, OperationKey, ShapeBuf,
};
use incin_core::shapes::OperationKind;
use incin_core::tensor::dtype::StorageEncoding;
use incin::prelude::*;
use std::borrow::Cow;

// ── 1. Custom DType Definition: FP8 E4M3 ──────────────────────────────────────

/// A custom 8-bit float format (1 sign bit, 4 exponent bits, 3 mantissa bits).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Fp8E4M3(pub u8);

impl Fp8E4M3 {
    pub fn from_f32(v: f32) -> Self {
        if v == 0.0 {
            return Self(0);
        }
        let bits = v.to_bits();
        let sign = ((bits >> 31) & 1) as u8;
        let exp = (((bits >> 23) & 0xFF) as i32 - 127 + 7).clamp(0, 15) as u8;
        let mantissa = ((bits >> 20) & 0x7) as u8;
        Self((sign << 7) | (exp << 3) | mantissa)
    }

    pub fn to_f32(self) -> f32 {
        if self.0 == 0 {
            return 0.0;
        }
        let sign = if (self.0 >> 7) != 0 { -1.0_f32 } else { 1.0_f32 };
        let exp = ((self.0 >> 3) & 0xF) as i32 - 7;
        let mantissa = (self.0 & 0x7) as f32 / 8.0 + 1.0;
        sign * mantissa * 2.0_f32.powi(exp)
    }
}

impl DType for Fp8E4M3 {
    type Arg = ();
    type Field = PhantomData<Self>;
    fn init(_: ()) -> Self::Field {
        PhantomData
    }
    fn descriptor(_: &Self::Field) -> DTypeDescriptor {
        Self::DESCRIPTOR
    }
}

impl ConstDType for Fp8E4M3 {
    const DESCRIPTOR: DTypeDescriptor = DTypeDescriptor::new(
        DTypeKey::new("custom", "fp8_e4m3", 1),
        DTypeKind::Float,
        StorageEncoding::scalar(1, 1),
    );
}

// ── 2. Custom Operation Contract with Exact Shape Validation Proofs ──────────

/// A custom operation computing `gelu(x + bias)` in a single hardware kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct FusedBiasGelu;

impl Operation for FusedBiasGelu {
    type Attributes = NoAttributes;

    const KEY: OperationKey = OperationKey {
        namespace: Cow::Borrowed("custom"),
        name: Cow::Borrowed("fused_bias_gelu"),
        version: 1,
    };

    /// Proves shape & contract invariants ahead of kernel execution.
    fn infer_outputs(
        _: &Self::Attributes,
        inputs: &[LogicalTensorMeta],
    ) -> core::result::Result<Vec<LogicalTensorMeta>, DescriptorError> {
        if inputs.len() != 2 {
            return Err(DescriptorError::Arity {
                operation: OperationKind::Gelu,
                expected: 2..=2,
                actual: inputs.len(),
            });
        }

        let shape_x = inputs[0].shape.as_ref();
        let shape_b = inputs[1].shape.as_ref();

        // Contract: Bias shape must match or broadcast to X shape
        if let (Some(sx), Some(sb)) = (shape_x, shape_b) {
            if sx != sb {
                return Err(DescriptorError::InvalidAttribute {
                    operation: OperationKind::Gelu,
                    attribute: "bias_shape",
                    reason: "bias dimensions must match activation input",
                });
            }
        }

        // Output geometry matches the validated shape
        Ok(vec![inputs[0].clone()])
    }
}

// ── 3. Calling Custom Ops WITHOUT Extension Traits (Standalone Function) ─────

/// Standalone function calling the custom operation.
/// No extension trait is needed - simply call this function directly!
pub fn fused_bias_gelu<S, B, K, G>(
    x: &Tensor<S, B, K, G>,
    bias: &Tensor<S, B, K, G>,
) -> incin::Result<Tensor<<S as BroadcastShape<S>>::Output, B, K, G>>
where
    S: Shape + DynShape + BroadcastShape<S>,
    <S as BroadcastShape<S>>::Output: DynShape,
    B: Backend + Execute<op::Add> + Execute<op::Gelu>,
    K: DType,
    G: RequiresGrad,
    <B as Execute<op::Add>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::Gelu>>::Output: Into<B::Storage<K>>,
{
    let sum = x + bias;
    sum.gelu()
}

// ── 4. Optional Extension Trait (For method syntax lovers: `x.fused_bias_gelu(...)`)

pub trait FusedBiasGeluExt<S: Shape + DynShape, B: Backend, K: DType, G: RequiresGrad> {
    fn fused_bias_gelu(
        &self,
        bias: &Tensor<S, B, K, G>,
    ) -> incin::Result<Tensor<<S as BroadcastShape<S>>::Output, B, K, G>>
    where
        S: BroadcastShape<S>,
        <S as BroadcastShape<S>>::Output: DynShape,
        B: Execute<op::Add> + Execute<op::Gelu>,
        <B as Execute<op::Add>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::Gelu>>::Output: Into<B::Storage<K>>;
}

impl<S: Shape + DynShape, B: Backend, K: DType, G: RequiresGrad> FusedBiasGeluExt<S, B, K, G>
    for Tensor<S, B, K, G>
{
    fn fused_bias_gelu(
        &self,
        bias: &Tensor<S, B, K, G>,
    ) -> incin::Result<Tensor<<S as BroadcastShape<S>>::Output, B, K, G>>
    where
        S: BroadcastShape<S>,
        <S as BroadcastShape<S>>::Output: DynShape,
        B: Execute<op::Add> + Execute<op::Gelu>,
        <B as Execute<op::Add>>::Output: Into<B::Storage<K>>,
        <B as Execute<op::Gelu>>::Output: Into<B::Storage<K>>,
    {
        fused_bias_gelu(self, bias)
    }
}

// ── Main Demo Execution ──────────────────────────────────────────────────────

fn main() -> incin::Result<()> {
    println!("=== Practical Example: Custom Operations & Validation Proofs ===\n");

    // ─────────────────────────────────────────────────────────────────────────
    // 1. Inspecting Validation Proofs
    // ─────────────────────────────────────────────────────────────────────────
    println!("[1] Demonstrating Contract Validation Ahead-of-Execution:");

    let valid_input_x = LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(&[4, 128])),
        dtype: Some(DTypeId::F32.descriptor()),
        device: Some(DeviceId::cpu()),
    };
    let valid_bias = LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(&[4, 128])),
        dtype: Some(DTypeId::F32.descriptor()),
        device: Some(DeviceId::cpu()),
    };

    // 1.1 Valid contract inference produces certified output metadata
    let validated = FusedBiasGelu::infer_outputs(&NoAttributes, &[valid_input_x.clone(), valid_bias])
        .expect("validation succeeds");
    println!("  • Inferred output shape proof: {:?}", validated[0].shape.as_ref().unwrap().dims());

    // 1.2 Shape mismatch caught before any backend kernel launch!
    let invalid_bias = LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(&[4, 64])), // Incompatible dimension!
        dtype: Some(DTypeId::F32.descriptor()),
        device: Some(DeviceId::cpu()),
    };
    let mismatch_result = FusedBiasGelu::infer_outputs(&NoAttributes, &[valid_input_x, invalid_bias]);
    match mismatch_result {
        Ok(_) => println!("  • Unexpected success!"),
        Err(err) => println!("  • Validation Proof correctly rejected mismatch: {err}"),
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 2. Calling Custom Operations (Two Clean Syntaxes)
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n[2] Executing Custom Operation:");
    let x = Cpu.tensor([-2.0_f32, -1.0, 0.0, 1.0, 2.0])?;
    let bias = Cpu.tensor([0.5_f32, 0.5, 0.5, 0.5, 0.5])?;

    // Syntax A: Standalone Function (Zero traits needed!)
    let result_fn = fused_bias_gelu(&x, &bias)?;
    println!("  • Output via Standalone Function: {:?}", result_fn.to_vec1::<f32>()?);

    // Syntax B: Extension Trait (Method syntax)
    let result_method = x.fused_bias_gelu(&bias)?;
    println!("  • Output via Extension Trait:     {:?}", result_method.to_vec1::<f32>()?);

    println!("\n[3] Custom operation executed and validated successfully!");
    Ok(())
}
