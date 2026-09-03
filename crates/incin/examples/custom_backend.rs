//! Practical Example: Authoring a Custom Compute Backend in 100% Safe Rust.
//!
//! Demonstrates:
//! 1. Defining a custom device type `MyCustomDevice`
//! 2. Defining custom tensor storage `MyStorage<K>` without any unsafe code
//! 3. Implementing the core backend contracts:
//!    - `StorageBackend`
//!    - `SupportsDType`
//!    - `HostReadback` & `HostInterop`
//!    - `VariableBackend`
//!    - `Backend`
//! 4. Implementing exact operation kernels via `Execute<Op>`:
//!    - `Execute<op::Zeros>`
//!    - `Execute<op::Add>`
//!    - `Execute<op::MatMulExact>`
//!    - `Execute<op::Relu>`
//! 5. Wiring the custom backend into Incin's `Target` API for `MyCustomDevice.zeros(...)`
//!
//! Run with: `cargo run -p incin --example custom_backend --features cpu,backend-authoring`

#![forbid(unsafe_code)]
#![allow(missing_docs)]
#![allow(clippy::type_complexity)]

use core::marker::PhantomData;
use incin::backend_authoring::operations::op;
use incin::backend_authoring::{
    Alignment, Capabilities, CapabilityQuery, Execute, ExecutionRequest, HostInterop, HostReadback,
    ShapeBuf, StorageBackend, StorageOutput, SupportLevel, SupportsDType, TensorMeta,
    VariableBackend,
};
use incin::prelude::*;
use incin_core::error::BackendError;
use std::sync::Arc;

// ── 1. Custom Device Definition ──────────────────────────────────────────────

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct MyCustomDevice;

impl Device for MyCustomDevice {
    type Arg = ();
    type Field = core::marker::PhantomData<Self>;

    fn init(_arg: Self::Arg) -> Self::Field {
        core::marker::PhantomData
    }

    fn to_incin(_dev: &Self::Field) -> incin::Result<DeviceId> {
        Ok(DeviceId::cpu())
    }
}

impl ConstDevice for MyCustomDevice {}

// ── 2. Custom Storage Type (100% Safe Rust) ──────────────────────────────────

#[derive(Debug, Clone)]
pub struct MyStorage<K = f32> {
    pub bytes: Arc<Vec<u8>>,
    pub meta: TensorMeta,
    pub _marker: PhantomData<K>,
}

impl<K: DType + ConstDType> MyStorage<K> {
    /// Constructs storage safely from float elements.
    pub fn from_floats(floats: &[f32], dims: &[usize]) -> Self {
        let mut bytes = Vec::with_capacity(floats.len() * 4);
        for &f in floats {
            bytes.extend_from_slice(&f.to_ne_bytes());
        }
        let meta = TensorMeta::contiguous(
            ShapeBuf::from_slice(dims),
            K::DESCRIPTOR,
            DeviceId::cpu(),
            Alignment::new(8).unwrap(),
            dims.iter().product(),
        )
        .expect("valid tensor metadata");

        Self {
            bytes: Arc::new(bytes),
            meta,
            _marker: PhantomData,
        }
    }

    /// Reads floats safely from byte storage.
    pub fn to_floats(&self) -> Vec<f32> {
        self.bytes
            .chunks_exact(4)
            .map(|chunk| {
                let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                f32::from_ne_bytes(arr)
            })
            .collect()
    }
}

impl<K: DType + 'static> StorageOutput for MyStorage<K> {}

// ── 3. Custom Backend Struct & Core Contracts ────────────────────────────────

#[derive(Clone, Copy, Debug, Default)]
pub struct MyCustomBackend;

impl StorageBackend for MyCustomBackend {
    const BACKEND_NAME: &'static str = "MyCustomBackend";
    type Device = MyCustomDevice;
    type Storage<K: DType> = MyStorage<K>;

    fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta {
        &storage.meta
    }

    fn execution_storage<K: DType>(
        storage: &Self::Storage<K>,
    ) -> (&dyn core::any::Any, Option<usize>)
    where
        Self::Storage<K>: core::any::Any,
    {
        (storage, None)
    }
}

impl<K: DType + ConstDType> SupportsDType<K> for MyCustomBackend {
    fn resolve_dtype(_field: &K::Field, _device: &DeviceId) -> incin::Result<DTypeDescriptor> {
        Ok(K::DESCRIPTOR)
    }
}

impl HostReadback for MyCustomBackend {
    fn float_to_vec1<K: DType>(storage: &Self::Storage<K>) -> incin::Result<Vec<f64>> {
        let floats: Vec<f64> = storage
            .bytes
            .chunks_exact(4)
            .map(|chunk| {
                let arr: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
                f32::from_ne_bytes(arr) as f64
            })
            .collect();
        Ok(floats)
    }

    fn int_to_vec1<K: DType>(storage: &Self::Storage<K>) -> incin::Result<Vec<i64>> {
        let ints: Vec<i64> = storage
            .bytes
            .chunks_exact(8)
            .map(|chunk| {
                let arr: [u8; 8] = [
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ];
                i64::from_ne_bytes(arr)
            })
            .collect();
        Ok(ints)
    }
}

impl HostInterop for MyCustomBackend {
    fn from_bytes<K: DType>(
        bytes: &[u8],
        dims: &[usize],
        _dtype: DTypeDescriptor,
        _device: &DeviceId,
    ) -> incin::Result<Self::Storage<K>> {
        let count: usize = dims.iter().product();
        let meta = TensorMeta::contiguous(
            ShapeBuf::from_slice(dims),
            _dtype,
            DeviceId::cpu(),
            Alignment::new(8).unwrap(),
            count,
        )
        .map_err(|e| incin::Error::Msg(format!("{:?}", e)))?;

        Ok(MyStorage {
            bytes: Arc::new(bytes.to_vec()),
            meta,
            _marker: PhantomData,
        })
    }

    fn to_bytes<K: DType>(storage: &Self::Storage<K>) -> incin::Result<Vec<u8>> {
        Ok(storage.bytes.as_ref().clone())
    }
}

#[derive(Debug, Clone)]
pub struct MyVar<K>(pub Arc<std::sync::RwLock<MyStorage<K>>>);

impl VariableBackend for MyCustomBackend {
    type Var<K: DType> = MyVar<K>;

    fn var_from_tensor<K: DType>(storage: &Self::Storage<K>) -> incin::Result<Self::Var<K>> {
        Ok(MyVar(Arc::new(std::sync::RwLock::new(storage.clone()))))
    }

    fn var_as_tensor<K: DType>(var: &Self::Var<K>) -> incin::Result<Self::Storage<K>> {
        Ok(var.0.read().unwrap().clone())
    }

    fn assign_var<K: DType>(var: &mut Self::Var<K>, value: &Self::Storage<K>) -> incin::Result<()> {
        *var.0.write().unwrap() = value.clone();
        Ok(())
    }
}

impl Capabilities for MyCustomBackend {
    fn support(&self, _query: &CapabilityQuery) -> SupportLevel {
        SupportLevel::Native
    }
}

impl incin::backend_authoring::Backend for MyCustomBackend {
    type InnerBackend = Self;
}

// ── 4. Operation Kernels (Execute<Op> in Safe Rust) ───────────────────────────

// op::Zeros Kernel
impl Execute<op::Zeros> for MyCustomBackend {
    type Output = MyStorage<f32>;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Zeros, Self>,
    ) -> core::result::Result<Self::Output, BackendError> {
        let attrs = request.operation.descriptor().attributes();
        let dims: &[usize] = attrs.shape.as_slice();
        let numel: usize = dims.iter().product();
        let zeros = vec![0.0_f32; numel];
        Ok(MyStorage::from_floats(&zeros, dims))
    }
}

// op::Add Kernel
impl Execute<op::Add> for MyCustomBackend {
    type Output = MyStorage<f32>;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Add, Self>,
    ) -> core::result::Result<Self::Output, BackendError> {
        let lhs: &MyStorage<f32> = request.inputs[0].downcast_ref().unwrap();
        let rhs: &MyStorage<f32> = request.inputs[1].downcast_ref().unwrap();

        let lhs_f = lhs.to_floats();
        let rhs_f = rhs.to_floats();
        let out: Vec<f32> = lhs_f.iter().zip(rhs_f.iter()).map(|(a, b)| a + b).collect();
        let dims: &[usize] = lhs.meta.shape().dims();
        Ok(MyStorage::from_floats(&out, dims))
    }
}

// op::MatMulExact Kernel
impl Execute<op::MatMulExact> for MyCustomBackend {
    type Output = MyStorage<f32>;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::MatMulExact, Self>,
    ) -> core::result::Result<Self::Output, BackendError> {
        let a: &MyStorage<f32> = request.inputs[0].downcast_ref().unwrap();
        let b: &MyStorage<f32> = request.inputs[1].downcast_ref().unwrap();

        let a_dims: &[usize] = a.meta.shape().dims();
        let b_dims: &[usize] = b.meta.shape().dims();
        let m = a_dims[0];
        let k = a_dims[1];
        let n = b_dims[1];

        let mut c = vec![0.0_f32; m * n];
        let a_data = a.to_floats();
        let b_data = b.to_floats();

        for i in 0..m {
            for p in 0..k {
                let a_ip = a_data[i * k + p];
                for j in 0..n {
                    c[i * n + j] += a_ip * b_data[p * n + j];
                }
            }
        }

        Ok(MyStorage::from_floats(&c, &[m, n]))
    }
}

// op::Relu Kernel
impl Execute<op::Relu> for MyCustomBackend {
    type Output = MyStorage<f32>;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::Relu, Self>,
    ) -> core::result::Result<Self::Output, BackendError> {
        let x: &MyStorage<f32> = request.inputs[0].downcast_ref().unwrap();
        let x_f = x.to_floats();
        let out: Vec<f32> = x_f.iter().map(|&v| if v > 0.0 { v } else { 0.0 }).collect();
        let dims: &[usize] = x.meta.shape().dims();
        Ok(MyStorage::from_floats(&out, dims))
    }
}

// ── 5. Target API Wiring ─────────────────────────────────────────────────────

impl incin_backends::target::TensorTarget for MyCustomDevice {
    type Dtype = f32;
    type ParameterDtype = f32;
    type Device = Self;
    type Backend = MyCustomBackend;

    fn device_arg(&self) {}
    fn dtype_field(&self) -> <Self::Dtype as DType>::Field {
        <Self::Dtype as DType>::init(())
    }
    fn parameter_dtype_field(&self) -> <Self::ParameterDtype as DType>::Field {
        <Self::ParameterDtype as DType>::init(())
    }
    fn precision_policy(&self) -> incin_backends::target::RuntimePrecisionPolicy {
        incin_backends::target::RuntimePrecisionPolicy::fp32()
    }
}

// ── Main Demo Execution ──────────────────────────────────────────────────────

fn main() -> incin::Result<()> {
    println!("=== Practical Example: Custom Backend Implementation (100% Safe Rust) ===");

    // 1. Allocate on custom backend via TargetExt
    println!("\n[1] Allocating zeros on MyCustomDevice target...");
    let zeros = MyCustomDevice.zeros(shape![2, 3])?;
    println!("  • Zeros allocated: shape {:?}", zeros.dims());

    // 2. Load host data onto custom backend
    println!("\n[2] Ingesting host matrix data onto MyCustomDevice...");
    let a = MyCustomDevice.tensor([[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]])?;
    let b = MyCustomDevice.tensor([[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0]])?;

    // 3. Execute matrix multiplication on custom backend
    println!("\n[3] Executing MatMul on MyCustomBackend...");
    let c = a.matmul(&b)?;
    println!("  • MatMul output shape: {:?}", c.dims());
    println!("  • Result data: {:?}", c.to_vec1::<f32>()?);

    // 4. Elementwise addition and ReLU
    println!("\n[4] Executing Add & ReLU on MyCustomBackend...");
    let d = MyCustomDevice.tensor([[-10.0_f32, 20.0], [30.0, -40.0]])?;
    let sum = &c + &d;
    let relu_out = sum.relu()?;
    println!("  • ReLU output data: {:?}", relu_out.to_vec1::<f32>()?);

    println!(
        "\n[5] Custom backend executed all tensor operations successfully without unsafe code!"
    );
    Ok(())
}
