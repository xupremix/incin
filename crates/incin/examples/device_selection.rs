//! The `B` parameter of `Tensor<S, B, K, G, P>`: the four ways to say which
//! device a tensor lives on, and which of them the compiler can check.
//!
//! Picking a device in incin is not one API but a ladder, and the rungs differ
//! in *when* the answer is known. That is the whole point: a backend chosen at
//! compile time lets `B: Execute<op::MatMul>` be checked before the program
//! runs, while a backend discovered at run time can only be checked when the
//! hardware is in front of you. Both are supported; they are not the same
//! guarantee, and this file makes the difference visible.
//!
//! The trap worth naming up front: `best_device!()` and `detect_device()` sound
//! like the same question and are not. `best_device!()` resolves against the
//! *features this build was compiled with* and touches no hardware at all.
//! `detect_device()` probes the machine. On a CPU-only build the first always
//! says CPU; on a `--features wgpu` build with no working adapter, the first
//! still says WGPU while the second says CPU.
//!
//! Run with: `cargo run -p incin --example device_selection --features cpu`

// These examples exist to spell out tensor types in full, so the type
// parameters are visible at the point of use rather than hidden behind an
// alias. That is exactly what `type_complexity` asks you to stop doing.
#![allow(clippy::type_complexity)]

use incin::prelude::*;

fn main() -> incin::Result<()> {
    section("1. Fully compile-time: the device is a type");
    fully_static()?;

    section("2. Compile-time family, compile-time ordinal");
    static_ordinal()?;

    section("3. Build-directed: best_device!()");
    build_directed()?;

    section("4. Run time: detect_device()");
    runtime_detected()?;

    Ok(())
}

/// The device is written into the tensor's type. Nothing about the placement
/// of this tensor is decided at run time, so an operation this backend cannot
/// execute is a compile error, not a `Result::Err`.
fn fully_static() -> incin::Result<()> {
    type Dev = IncinBackend<Cpu>;

    let tensor: Tensor<s![3, 3], Dev> = Tensor::zeros(())?;
    let activated = tensor.relu()?;

    println!("  IncinBackend<Cpu>, shape {:?}", activated.dims());
    println!("  the device is part of the type, so it costs nothing at run time");
    Ok(())
}

/// The same, with the ordinal also fixed at compile time. `WgpuN<U0>` names
/// adapter 0 specifically, so two tensors on different adapters have different
/// types and cannot be mixed by accident.
///
/// Spelled with the CPU here so this example runs on a CPU-only build; the
/// accelerator form is identical with `WgpuN<U0>` or `CudaN<U1>` in place of
/// `Cpu`, and `crates/incin/examples/backends/` shows the WGPU version.
fn static_ordinal() -> incin::Result<()> {
    type Dev = IncinBackend<Cpu>;

    let tensor: Tensor<s![2, 4], Dev> = Tensor::ones(())?;
    println!("  IncinBackend<Cpu>, shape {:?}", tensor.dims());
    println!("  on an accelerator build this reads IncinBackend<WgpuN<U0>>");
    Ok(())
}

/// `best_device!()` names the most capable device *this build can target*.
///
/// It performs no discovery. It expands to a type alias resolved from the
/// feature flags incin itself was compiled with, which is why it can appear in
/// a type position at all. A machine with four idle GPUs and a CPU-only build
/// gets CPU here, and that is correct: the build has no code that could target
/// the GPUs.
fn build_directed() -> incin::Result<()> {
    type Dev = IncinBackend<incin_core::best_device!()>;

    let tensor: Tensor<s![2, 2], Dev> = Tensor::zeros(())?;
    println!("  best_device!() resolved to {:?}", tensor.device()?);
    println!("  no hardware was probed to produce this type");
    Ok(())
}

/// `detect_device()` probes the machine and returns a `DeviceId`, a run-time
/// value. It tries CUDA, then Metal, then WGPU, then CPU, and returns the
/// first family with usable hardware in this build.
///
/// Because the answer is a value rather than a type, the tensor that uses it
/// has to be dynamic in its backend: `IncinBackend<Dyn>` accepts the device as
/// a constructor argument. That is the trade. The program now runs on whatever
/// is present, and in exchange the compiler can no longer tell you that this
/// device cannot do this operation.
fn runtime_detected() -> incin::Result<()> {
    let Some(device) = incin_backends::detect_device() else {
        println!("  no usable backend detected in this build");
        return Ok(());
    };

    let tensor: Tensor<Dyn, IncinBackend<Dyn>> = Tensor::zeros(([2, 3], device))?;
    println!("  detect_device() probed the machine and chose {device:?}");
    println!("  shape {:?}, backend known only at run time", tensor.dims());

    // A caller that wants a policy rather than the default preference order
    // can pin one. This refuses CUDA even where it is present.
    let pinned = incin_backends::detect_device_in(&[
        incin_core::tensor::device::DeviceKind::Wgpu,
        incin_core::tensor::device::DeviceKind::Cpu,
    ]);
    println!("  a Wgpu-then-Cpu policy would choose {pinned:?}");

    // The two questions, side by side. They agree whenever the most capable
    // compiled-in backend also has working hardware, and disagree the moment
    // it does not: a `--features cuda` build on a machine with no NVIDIA card
    // still resolves `best_device!()` to CUDA while `detect_device()` falls
    // through to whatever is actually present.
    let build_choice: Tensor<s![1], IncinBackend<incin_core::best_device!()>> =
        Tensor::zeros(())?;
    println!();
    println!("  build says   {:?}", build_choice.device()?);
    println!("  hardware says {device:?}");

    Ok(())
}

fn section(title: &str) {
    println!("\n{title}");
    println!("{}", "-".repeat(title.len()));
}
