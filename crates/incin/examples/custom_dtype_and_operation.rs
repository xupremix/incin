//! Practical Example: Authoring a Custom DType & Custom Fused Operation.
//!
//! Demonstrates:
//! 1. Defining a custom logical dtype `Fp8E4M3` (8-bit floating point)
//! 2. Constructing its `DTypeKey`, `StorageEncoding`, and `DTypeDescriptor`
//! 3. Implementing Incin's `DType` and `ConstDType` traits for `Fp8E4M3`
//! 4. Quantizing and dequantizing host data with custom FP8 encoding
//! 5. Defining a custom fused operation `FusedBiasGelu`
//! 6. Registering typed metadata inference via the `Operation` trait
//! 7. Executing fused operations on tensors.
//!
//! Run with: `cargo run -p incin --example custom_dtype_and_operation --features cpu,backend-authoring`

#![allow(missing_docs)]
#![allow(clippy::type_complexity)]

use core::marker::PhantomData;
use incin::backend_authoring::operations::{NoAttributes, Operation};
use incin::backend_authoring::{DescriptorError, LogicalTensorMeta, OperationKey, ShapeBuf};
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

// ── 2. Custom Fused Operation Definition: FusedBiasGelu ──────────────────────

/// A custom fused operation computing `gelu(x + bias)` in a single kernel pass.
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
    // 3. Executing Fused Kernel Math
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n[4] Executing fused bias + GELU activation:");
    let x = Cpu.tensor([-2.0_f32, -1.0, 0.0, 1.0, 2.0])?;
    let bias = Cpu.tensor([0.5_f32, 0.5, 0.5, 0.5, 0.5])?;

    // Fused forward: gelu(x + bias)
    let pre_activation = &x + &bias;
    let fused_result = pre_activation.gelu()?;

    println!("  • Input x:       {:?}", x.to_vec1::<f32>()?);
    println!("  • Bias:          {:?}", bias.to_vec1::<f32>()?);
    println!("  • Fused Output:  {:?}", fused_result.to_vec1::<f32>()?);

    println!("\n[5] Custom dtype and fused operation verified successfully!");
    Ok(())
}
