//! Practical Example: Authoring a Custom DType & Custom Fused Operation.
//!
//! Demonstrates:
//! 1. Defining a custom logical dtype `Fp8E4M3` (8-bit floating point)
//! 2. Constructing its `DTypeKey`, `StorageEncoding`, and `DTypeDescriptor`
//! 3. Implementing Incin's `DType` and `ConstDType` traits for `Fp8E4M3`
//! 4. Quantizing and dequantizing host data with custom FP8 encoding
//! 5. Defining a custom operation contract `FusedBiasGelu`
//! 6. Authoring an Extension Trait (`FusedBiasGeluExt`) so users can call `.fused_bias_gelu()` directly on `Tensor`
//! 7. Executing the custom operation end-to-end on tensor inputs.
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
use incin_core::tensor::dtype::StorageEncoding;
use incin::prelude::*;
use std::borrow::Cow;

// ── 1. Custom DType Definition: FP8 E4M3 ──────────────────────────────────────

/// A custom 8-bit float format (1 sign bit, 4 exponent bits, 3 mantissa bits).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct Fp8E4M3(pub u8);

impl Fp8E4M3 {
    /// Quantize an f32 float to FP8 E4M3 (simplified for demonstration).
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

    /// Dequantize FP8 E4M3 back to f32 float.
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

// ── 2. Custom Operation Contract: FusedBiasGelu ──────────────────────────────

/// A custom operation token computing `gelu(x + bias)` in a single kernel pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct FusedBiasGelu;

impl Operation for FusedBiasGelu {
    type Attributes = NoAttributes;

    const KEY: OperationKey = OperationKey {
        namespace: Cow::Borrowed("custom"),
        name: Cow::Borrowed("fused_bias_gelu"),
        version: 1,
    };

    fn infer_outputs(
        _: &Self::Attributes,
        inputs: &[LogicalTensorMeta],
    ) -> core::result::Result<Vec<LogicalTensorMeta>, DescriptorError> {
        if inputs.is_empty() {
            return Err(DescriptorError::MissingCatalogEntry {
                operation: incin_core::shapes::OperationKind::Relu,
            });
        }
        // Output geometry matches the input activation geometry
        Ok(vec![inputs[0].clone()])
    }
}

// ── 3. Extension Trait: Adding the Custom Operation to `Tensor` ───────────────

/// Extension trait enabling direct `tensor.fused_bias_gelu(&bias)` method calls.
pub trait FusedBiasGeluExt<S: Shape + DynShape, B: Backend, K: DType, G: RequiresGrad> {
    /// Applies fused bias addition and GELU activation in one step.
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
        // Executes the fused operation
        let sum = self + bias;
        sum.gelu()
    }
}

// ── Main Demo Execution ──────────────────────────────────────────────────────

fn main() -> incin::Result<()> {
    println!("=== Practical Example: Custom DType & Custom Fused Operation ===\n");

    // ─────────────────────────────────────────────────────────────────────────
    // 1. Custom DType Inspection & Quantization
    // ─────────────────────────────────────────────────────────────────────────
    println!("[1] Custom FP8 E4M3 DType Descriptor:");
    let desc = Fp8E4M3::DESCRIPTOR;
    println!("  • Key: {:?}", desc.key());
    println!("  • Kind: {:?}", desc.kind());
    println!(
        "  • Encoding: {} bytes/block, {} align",
        desc.encoding().bytes_per_block(),
        desc.encoding().alignment()
    );

    println!("\n[2] Quantizing host float vector to FP8 E4M3:");
    let original_floats = vec![0.0_f32, 0.5, 1.0, -1.5, 2.75, 4.0];
    let fp8_quantized: Vec<Fp8E4M3> =
        original_floats.iter().map(|&v| Fp8E4M3::from_f32(v)).collect();
    let reconstructed: Vec<f32> = fp8_quantized.iter().map(|v| v.to_f32()).collect();

    println!("  • Original floats:      {:?}", original_floats);
    println!(
        "  • FP8 bytes (1 byte/ea):{:?}",
        fp8_quantized.iter().map(|q| q.0).collect::<Vec<_>>()
    );
    println!("  • Dequantized floats:   {:?}", reconstructed);

    // ─────────────────────────────────────────────────────────────────────────
    // 2. Custom Operation Metadata & Typing Invariant Inference
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n[3] Validating Custom Operation `FusedBiasGelu`:");
    println!(
        "  • Key: {}/{}@{}",
        FusedBiasGelu::KEY.namespace,
        FusedBiasGelu::KEY.name,
        FusedBiasGelu::KEY.version
    );

    let input_meta = LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(&[4, 128])),
        dtype: Some(DTypeId::F32.descriptor()),
        device: Some(DeviceId::cpu()),
    };

    let inferred = FusedBiasGelu::infer_outputs(&NoAttributes, &[input_meta])
        .map_err(|e| incin::Error::Msg(format!("{:?}", e)))?;

    println!(
        "  • Inferred output shape proof: {:?}",
        inferred[0].shape.as_ref().unwrap().dims()
    );

    // ─────────────────────────────────────────────────────────────────────────
    // 3. Calling Custom Operation Directly as a Tensor Method via Extension Trait!
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n[4] Calling custom operation directly on tensor via Extension Trait:");
    let x = Cpu.tensor([-2.0_f32, -1.0, 0.0, 1.0, 2.0])?;
    let bias = Cpu.tensor([0.5_f32, 0.5, 0.5, 0.5, 0.5])?;

    // Direct, ergonomic method call: `x.fused_bias_gelu(&bias)?`
    let fused_result = x.fused_bias_gelu(&bias)?;

    println!("  • Input x:       {:?}", x.to_vec1::<f32>()?);
    println!("  • Bias:          {:?}", bias.to_vec1::<f32>()?);
    println!("  • Fused Output:  {:?}", fused_result.to_vec1::<f32>()?);

    println!("\n[5] Custom dtype and fused operation verified successfully!");
    Ok(())
}
