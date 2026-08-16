//! The `target-api` prototype against a real accelerator.
//!
//! `Wgpu` is a Tier 2 device: the backend family is fixed at compile time and
//! the ordinal is a runtime value. That is the case the current constructor
//! surface handles worst — it needs `((), Wgpu::new(0))`, a 2-tuple whose
//! leading unit exists only to satisfy `ArgInto`'s slot bookkeeping. Here the
//! device value *is* the target.
//!
//! Execution-verified rather than compile-checked: this environment has a
//! software adapter.

#![cfg(feature = "wgpu")]

extern crate incin_core as incin;

use incin_backends::prelude::*;
use incin_core::prelude::*;

#[test]
fn a_tier_two_device_value_is_a_target_with_no_extra_ceremony() {
    let gpu = Wgpu::new(0);
    assert_eq!(gpu.device_id().unwrap(), DeviceId::wgpu(0));

    let x = gpu.zeros(shape![2, 3]).unwrap();
    assert_eq!(x.dims(), [2, 3]);
    // The tensor really landed on the wgpu device, not silently on the CPU.
    assert!(x.to_string().contains("device=wgpu:0"), "got {x}");
}

#[test]
fn typed_data_reaches_the_accelerator_with_its_own_dtype() {
    let gpu = Wgpu::new(0);
    let x = gpu.tensor([[1.0_f32, 2.0], [3.0, 4.0]]).unwrap();
    assert_eq!(x.dims(), [2, 2]);
    assert_eq!(x.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn a_partial_shape_survives_onto_the_accelerator() {
    let gpu = Wgpu::new(0);
    let batch = 3usize;
    let x = gpu.zeros(shape![batch, 4]).unwrap();
    assert_eq!(x.dims(), [3, 4]);
}

#[test]
fn with_dtype_refuses_a_dtype_the_accelerator_cannot_store() {
    let gpu = Wgpu::new(0);
    let device = gpu.device_id().unwrap();
    let refused =
        <incin_backends::wgpu::WgpuBackendImpl<Wgpu> as SupportsDType<Dyn>>::resolve_dtype(
            &DTypeId::F64.into(),
            &device,
        )
        .expect_err("the wgpu storage row lists f32 only");
    let message = refused.to_string();
    assert!(message.contains("unsupported by backend 'Wgpu' for 'dtype'"));
    assert!(message.contains("f64"));

    assert!(gpu.dtype::<f32>().is_ok());
}
