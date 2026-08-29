//! Target API with runtime-known devices, dynamic dtypes, and precision policies.
//!
//! While Incin offers compile-time static types (`Cpu`, `CudaN<U0>`, `f32`, `s![2, 3]`),
//! production ML pipelines frequently receive target configurations, device indices,
//! dtypes, and batch dimensions from command-line arguments, JSON config files, or
//! hardware probes at runtime.
//!
//! This example shows how `Target<E, D, P>` and `TargetExt` handle runtime-known:
//! 1. Devices: dynamically probed (`detect_device()`) or constructed (`DeviceId`)
//! 2. DTypes: dynamically rebound via `.dtype_dynamic(descriptor)`
//! 3. Geometries: dynamic runtime shapes (`[batch, 784]`, `vec![...]`)
//! 4. Precision: mixed-precision AMP policies (`RuntimePrecisionPolicy::mixed_bf16()`)
//!
//! Run with: `cargo run -p incin --example target_api_dynamic --features cpu`

#![allow(clippy::type_complexity)]

use incin::prelude::*;
use incin_backends::detect::detect_device;
use incin_backends::target::{Native, RuntimePrecisionPolicy, Target};
use incin_core::tensor::device::DeviceId;
use incin_core::tensor::dtype::{DTypeId, DTypeDescriptor};

fn main() -> incin::Result<()> {
    section("1. Runtime Device Detection & Target Creation");
    runtime_device_target()?;

    section("2. Dynamic DType Rebinding (.dtype_dynamic)");
    dynamic_dtype_rebinding()?;

    section("3. Runtime Data Ingestion & Shape Allocation");
    data_ingestion()?;

    section("4. Runtime Precision Policy (Automatic Mixed Precision)");
    runtime_precision_policy()?;

    Ok(())
}

/// 1. Creating a target when the physical device is resolved at runtime.
fn runtime_device_target() -> incin::Result<()> {
    // Probe physical hardware on this machine (tries CUDA -> Metal -> WGPU -> CPU).
    let device_id: DeviceId = detect_device().unwrap_or_else(DeviceId::cpu);
    println!("  [+] Discovered active host device: {:?}", device_id);

    // Construct a Target value with the Native engine and the discovered DeviceId.
    let target: Target<Native, Dyn> = Target::new((), device_id, ());

    // Dynamic runtime dimensions (e.g. dynamic batch and sequence length)
    let batch_size = 4;
    let hidden_dim = 128;
    let activations = target.zeros([batch_size, hidden_dim])?;

    println!(
        "  [+] Allocated zero tensor on {:?}, dims: {:?}",
        target.device_id()?,
        activations.dims()
    );

    // Initializing neural network layers on the runtime target:
    let linear = incin::nn::linear::linear(shape![128, 64]).init(&target)?;
    println!(
        "  [+] Initialized linear layer weights on target: {:?}",
        linear.weight.shape_dims()
    );

    Ok(())
}

/// 2. Rebinding the target's generation dtype at runtime.
fn dynamic_dtype_rebinding() -> incin::Result<()> {
    let target: Target<Native, Dyn> = Target::new((), DeviceId::cpu(), ());

    // DType descriptor parsed from runtime config (e.g., "float64" or "int64")
    let f64_desc: DTypeDescriptor = DTypeId::F64.descriptor();
    let i64_desc: DTypeDescriptor = DTypeId::I64.descriptor();

    println!("  [+] Rebinding target dynamically to descriptor: {:?}", f64_desc);
    let f64_target = target.dtype_dynamic(f64_desc)?;
    let double_tensor: Tensor<Dyn, _, Dyn, NoGrad> = f64_target.ones([3, 3])?;
    println!(
        "  [+] Created tensor with dynamic dtype: {:?}",
        double_tensor.dims()
    );

    println!("  [+] Rebinding target dynamically to integer descriptor: {:?}", i64_desc);
    let i64_target = target.dtype_dynamic(i64_desc)?;
    let int_tensor: Tensor<Dyn, _, Dyn, NoGrad> = i64_target.zeros([5])?;
    println!(
        "  [+] Created integer tensor with dynamic dtype: {:?}",
        int_tensor.dims()
    );

    Ok(())
}

/// 3. Data ingestion where element dtypes are strictly preserved without casting.
fn data_ingestion() -> incin::Result<()> {
    let target: Target<Native, Dyn> = Target::new((), DeviceId::cpu(), ());

    // Target default float is f32, but integer label data is never silently cast:
    let labels = target.tensor([10_i64, 20, 30])?;
    assert_eq!(labels.to_vec1::<i64>()?, vec![10, 20, 30]);
    println!("  [+] Target::tensor preserved i64 labels unchanged: {:?}", labels.dims());

    // Creating tensors from dynamically-sized runtime vectors:
    let dynamic_values: Vec<f32> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let tensor = target.tensor_from_vec(dynamic_values, [2, 3])?;
    println!("  [+] Target::tensor_from_vec constructed [2, 3] tensor from runtime vector");
    assert_eq!(tensor.dims().as_ref(), &[2, 3]);

    Ok(())
}

/// 4. Decoupling activation and parameter storage with dynamic precision policies.
fn runtime_precision_policy() -> incin::Result<()> {
    let target: Target<Native, Dyn> = Target::new((), DeviceId::cpu(), ());

    // Configure mixed precision: compute in BF16 / FP16, store parameters in FP32
    let amp_policy = RuntimePrecisionPolicy::mixed_bf16();
    let mixed_target = target.with_runtime_precision(amp_policy);

    println!(
        "  [+] Target with dynamic precision policy: {:?}",
        mixed_target.precision_policy()
    );

    Ok(())
}

fn section(title: &str) {
    println!("\n{title}");
    println!("{}", "-".repeat(title.len()));
}
