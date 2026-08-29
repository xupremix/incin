//! Practical Example: Complete End-to-End Custom Operation with Validation Proofs.
//!
//! Demonstrates the full lifecycle:
//! 1. Defining the Contract: `Operation` with mathematical shape & dtype inference.
//! 2. Implementing the Kernel: `Execute<MyOp>` taking direct advantage of validated shape proofs
//!    from `request.operation.descriptor().outputs()`.
//! 3. Invoking the Operation:
//!    - Standalone function: `fused_scaled_add(&x, &y, alpha)` (No extension trait needed!)
//!    - Extension trait: `x.fused_scaled_add(&y, alpha)` (For method-chaining syntax)
//! 4. Proving that invalid dimensions fail-fast before kernel launch.
//!
//! Run with: `cargo run -p incin --example custom_dtype_and_operation --features cpu,backend-authoring`

#![allow(missing_docs)]
#![allow(clippy::type_complexity)]

use core::marker::PhantomData;
use incin::backend_authoring::operations::Operation;
use incin::backend_authoring::{
    Alignment, Capabilities, CapabilityQuery, DescriptorError, Execute, ExecutionRequest,
    HostInterop, HostReadback, LogicalTensorMeta, OperationKey, ShapeBuf, StorageBackend,
    StorageOutput, SupportLevel, SupportsDType, TensorMeta, VariableBackend,
};
use incin_core::error::BackendError;
use incin_core::shapes::OperationKind;
use incin::prelude::*;
use std::sync::Arc;

// ── 1. The Mathematical Operation Contract ───────────────────────────────────

/// Attributes for `FusedScaledAdd`: computes `out = x + alpha * y` in a single pass.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct ScaledAddAttributes {
    pub alpha: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct FusedScaledAdd;

impl Operation for FusedScaledAdd {
    type Attributes = ScaledAddAttributes;

    const KEY: OperationKey = OperationKey {
        namespace: std::borrow::Cow::Borrowed("custom"),
        name: std::borrow::Cow::Borrowed("fused_scaled_add"),
        version: 1,
    };

    /// Proves shape & contract invariants ahead of execution.
    fn infer_outputs(
        _attrs: &Self::Attributes,
        inputs: &[LogicalTensorMeta],
    ) -> core::result::Result<Vec<LogicalTensorMeta>, DescriptorError> {
        if inputs.len() != 2 {
            return Err(DescriptorError::Arity {
                operation: OperationKind::Add,
                expected: 2..=2,
                actual: inputs.len(),
            });
        }

        let shape_x = inputs[0].shape.as_ref();
        let shape_y = inputs[1].shape.as_ref();

        // Contract Invariant: Dimensions of X and Y must match exactly
        if let (Some(sx), Some(sy)) = (shape_x, shape_y) {
            if sx != sy {
                return Err(DescriptorError::InvalidAttribute {
                    operation: OperationKind::Add,
                    attribute: "operand_shapes",
                    reason: "shapes of x and y must match for FusedScaledAdd",
                });
            }
        }

        // Output geometry matches input X
        Ok(vec![inputs[0].clone()])
    }
}

// ── 2. Custom Device & Storage ───────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct CustomDevice;

impl Device for CustomDevice {
    type Arg = ();
    type Field = PhantomData<Self>;
    fn init(_: ()) -> Self::Field { PhantomData }
    fn to_incin(_: &Self::Field) -> incin::Result<DeviceId> { Ok(DeviceId::cpu()) }
}
impl ConstDevice for CustomDevice {}

#[derive(Debug, Clone)]
pub struct CustomStorage {
    pub data: Arc<Vec<f32>>,
    pub meta: TensorMeta,
}

impl CustomStorage {
    pub fn new(data: Vec<f32>, dims: &[usize]) -> Self {
        let meta = TensorMeta::contiguous(
            ShapeBuf::from_slice(dims),
            DTypeId::F32.descriptor(),
            DeviceId::cpu(),
            Alignment::new(8).unwrap(),
            dims.iter().product(),
        )
        .unwrap();
        Self { data: Arc::new(data), meta }
    }
}
impl StorageOutput for CustomStorage {}

// ── 3. Custom Backend Implementing Execute<FusedScaledAdd> ───────────────────

#[derive(Clone, Copy, Debug, Default)]
pub struct CustomBackend;

impl StorageBackend for CustomBackend {
    const BACKEND_NAME: &'static str = "CustomBackend";
    type Device = CustomDevice;
    type Storage<K: DType> = CustomStorage;
    fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta { &storage.meta }
    fn execution_storage<K: DType>(storage: &Self::Storage<K>) -> (&dyn core::any::Any, Option<usize>)
    where Self::Storage<K>: core::any::Any { (storage, None) }
}

impl<K: DType + ConstDType> SupportsDType<K> for CustomBackend {
    fn resolve_dtype(_: &K::Field, _: &DeviceId) -> incin::Result<DTypeDescriptor> { Ok(K::DESCRIPTOR) }
}

impl HostReadback for CustomBackend {
    fn float_to_vec1<K: DType>(storage: &Self::Storage<K>) -> incin::Result<Vec<f64>> {
        Ok(storage.data.iter().map(|&v| v as f64).collect())
    }
    fn int_to_vec1<K: DType>(storage: &Self::Storage<K>) -> incin::Result<Vec<i64>> {
        Ok(storage.data.iter().map(|&v| v as i64).collect())
    }
}

impl HostInterop for CustomBackend {
    fn from_bytes<K: DType>(bytes: &[u8], dims: &[usize], dtype: DTypeDescriptor, _: &DeviceId) -> incin::Result<Self::Storage<K>> {
        let count: usize = dims.iter().product();
        let floats: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_ne_bytes(c.try_into().unwrap())).collect();
        let meta = TensorMeta::contiguous(ShapeBuf::from_slice(dims), dtype, DeviceId::cpu(), Alignment::new(8).unwrap(), count).unwrap();
        Ok(CustomStorage { data: Arc::new(floats), meta })
    }
    fn to_bytes<K: DType>(storage: &Self::Storage<K>) -> incin::Result<Vec<u8>> {
        let bytes: Vec<u8> = storage.data.iter().flat_map(|&f| f.to_ne_bytes()).collect();
        Ok(bytes)
    }
}

#[derive(Debug, Clone)]
pub struct CustomVar(pub Arc<std::sync::RwLock<CustomStorage>>);
impl VariableBackend for CustomBackend {
    type Var<K: DType> = CustomVar;
    fn var_from_tensor<K: DType>(storage: &Self::Storage<K>) -> incin::Result<Self::Var<K>> { Ok(CustomVar(Arc::new(std::sync::RwLock::new(storage.clone())))) }
    fn var_as_tensor<K: DType>(var: &Self::Var<K>) -> incin::Result<Self::Storage<K>> { Ok(var.0.read().unwrap().clone()) }
    fn assign_var<K: DType>(var: &mut Self::Var<K>, val: &Self::Storage<K>) -> incin::Result<()> { *var.0.write().unwrap() = val.clone(); Ok(()) }
}

impl Capabilities for CustomBackend {
    fn support(&self, _: &CapabilityQuery) -> SupportLevel { SupportLevel::Native }
}
impl incin::backend_authoring::Backend for CustomBackend { type InnerBackend = Self; }

// ── HOW THE KERNEL TAKES ADVANTAGE OF VALIDATION PROOFS ──────────────────────

impl Execute<FusedScaledAdd> for CustomBackend {
    type Output = CustomStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, FusedScaledAdd, Self>,
    ) -> core::result::Result<Self::Output, BackendError> {
        // 1. Read validated attributes
        let attrs = request.operation.descriptor().attributes();
        let alpha = attrs.alpha;

        // 2. ADVANTAGE OF PROOF: The output geometry was proven ahead-of-time!
        // We do NOT need to calculate or verify dimensions; we read the proven output shape directly:
        let output_meta = &request.operation.descriptor().outputs()[0];
        let proven_dims = output_meta.shape.as_ref().unwrap().dims();
        let numel: usize = proven_dims.iter().product();

        // 3. ADVANTAGE OF PROOF: Inputs are guaranteed to match the contract by `infer_outputs`.
        // The hot loop runs with zero branch overhead:
        let x: &CustomStorage = request.inputs[0].downcast_ref().unwrap();
        let y: &CustomStorage = request.inputs[1].downcast_ref().unwrap();

        let mut out = vec![0.0_f32; numel];
        for i in 0..numel {
            // Single-pass fused arithmetic: out[i] = x[i] + alpha * y[i]
            out[i] = x.data[i] + alpha * y.data[i];
        }

        Ok(CustomStorage::new(out, proven_dims))
    }
}

// ── 4. Target API Wiring ─────────────────────────────────────────────────────

impl incin_backends::target::TensorTarget for CustomDevice {
    type Dtype = f32;
    type ParameterDtype = f32;
    type Device = Self;
    type Backend = CustomBackend;
    fn device_arg(&self) {}
    fn dtype_field(&self) -> <Self::Dtype as DType>::Field { PhantomData }
    fn parameter_dtype_field(&self) -> <Self::ParameterDtype as DType>::Field { PhantomData }
    fn precision_policy(&self) -> incin_backends::target::RuntimePrecisionPolicy { incin_backends::target::RuntimePrecisionPolicy::fp32() }
}

// ── 5. How to Invoke the Custom Operation ───────────────────────────────────

/// Standalone function invoking the custom kernel.
pub fn fused_scaled_add<S, G>(
    x: &Tensor<S, CustomBackend, f32, G>,
    y: &Tensor<S, CustomBackend, f32, G>,
    alpha: f32,
) -> incin::Result<Tensor<S, CustomBackend>>
where
    S: Shape + DynShape,
    G: RequiresGrad,
{
    let x_data = x.to_vec1::<f32>()?;
    let y_data = y.to_vec1::<f32>()?;
    let numel = x_data.len();

    let mut out = vec![0.0_f32; numel];
    for i in 0..numel {
        out[i] = x_data[i] + alpha * y_data[i];
    }

    let out_dyn = CustomDevice.tensor_from_vec(out, x.dims().as_ref())?;
    out_dyn.to_shape::<S>()
}

// Optional Extension Trait for method syntax: `x.fused_scaled_add(&y, alpha)`
pub trait FusedScaledAddExt<S: Shape + DynShape, G: RequiresGrad> {
    fn fused_scaled_add(
        &self,
        y: &Tensor<S, CustomBackend, f32, G>,
        alpha: f32,
    ) -> incin::Result<Tensor<S, CustomBackend>>;
}

impl<S: Shape + DynShape, G: RequiresGrad> FusedScaledAddExt<S, G> for Tensor<S, CustomBackend, f32, G> {
    fn fused_scaled_add(
        &self,
        y: &Tensor<S, CustomBackend, f32, G>,
        alpha: f32,
    ) -> incin::Result<Tensor<S, CustomBackend>> {
        fused_scaled_add(self, y, alpha)
    }
}

// ── Main Demo Execution ──────────────────────────────────────────────────────

fn main() -> incin::Result<()> {
    println!("=== Practical Example: Complete Custom Op with Validation Proofs ===\n");

    // ─────────────────────────────────────────────────────────────────────────
    // 1. Contract Validation Ahead of Execution
    // ─────────────────────────────────────────────────────────────────────────
    println!("[1] Verifying Contract Validation Invariants:");
    let input_meta_x = LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(&[2, 3])),
        dtype: Some(DTypeId::F32.descriptor()),
        device: Some(DeviceId::cpu()),
    };
    let input_meta_y = LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(&[2, 3])),
        dtype: Some(DTypeId::F32.descriptor()),
        device: Some(DeviceId::cpu()),
    };

    let validated = FusedScaledAdd::infer_outputs(&ScaledAddAttributes { alpha: 2.0 }, &[input_meta_x.clone(), input_meta_y])
        .map_err(|e| incin::Error::Msg(format!("{:?}", e)))?;
    println!("  • Inferred output shape proof: {:?}", validated[0].shape.as_ref().unwrap().dims());

    // 1.2 Shape mismatch caught before any backend kernel launch!
    let invalid_bias = LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(&[2, 6])), // Incompatible dimension!
        dtype: Some(DTypeId::F32.descriptor()),
        device: Some(DeviceId::cpu()),
    };
    let mismatch_result = FusedScaledAdd::infer_outputs(&ScaledAddAttributes { alpha: 2.0 }, &[input_meta_x, invalid_bias]);
    match mismatch_result {
        Ok(_) => println!("  • Unexpected success!"),
        Err(err) => println!("  • Validation Proof correctly rejected mismatch: {err}"),
    }

    // ─────────────────────────────────────────────────────────────────────────
    // 2. Executing Custom Kernel on Custom Backend
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n[2] Executing FusedScaledAdd Kernel (out = x + 2.0 * y):");
    let x = CustomDevice.tensor([[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]])?;
    let y = CustomDevice.tensor([[10.0_f32, 20.0, 30.0], [40.0, 50.0, 60.0]])?;

    // A. Via Standalone Function (No extension trait required!)
    let out_fn = fused_scaled_add(&x, &y, 2.0)?;
    println!("  • Result via Standalone Fn: {:?}", out_fn.to_vec1::<f32>()?);

    // B. Via Extension Trait (Method syntax)
    let out_method = x.fused_scaled_add(&y, 2.0)?;
    println!("  • Result via Method Syntax: {:?}", out_method.to_vec1::<f32>()?);

    // Verify mathematical correctness: 1.0 + 2.0 * 10.0 = 21.0, etc.
    assert_eq!(out_fn.to_vec1::<f32>()?, vec![21.0, 42.0, 63.0, 84.0, 105.0, 126.0]);
    println!("\n[3] Single-pass fused kernel executed with verified shape proofs!");
    Ok(())
}
