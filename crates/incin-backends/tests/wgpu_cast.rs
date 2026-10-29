//! `to_dtype` on WGPU through the user-facing `Tensor` surface (#91).
//!
//! `quantize`/`dequantize` stay refused on this backend (no `Q8_0` buffer
//! representation exists here — see `wgpu/backend/convert.rs`' module docs
//! for the investigated-deferred reason), but elementwise casts between the
//! storage dtypes WGPU can honestly hold now execute: `f32` sources cast
//! host-side to `{f32,u8,u32,i64}`, matching CPU value-for-value (Rust `as`
//! truncation both sides). Targets outside that set — `f64`, `bool`,
//! `Q8_0` — are refused by the executor, which is load-bearing because
//! capability admission never inspects the target tag.
//!
//! Requires a WGPU adapter:
//! `cargo test -p incin-backends --no-default-features --features std,cpu,wgpu --test wgpu_cast`.
#![cfg(all(feature = "wgpu", feature = "cpu"))]

extern crate incin_core as incin;

use incin_backends::prelude::*;
use incin_core::prelude::*;
use incin_core::tensor::device::Cpu;

fn require_wgpu(gpu: &Wgpu) {
    gpu.zeros(shape![1]).expect(
        "no WGPU adapter, but the `wgpu` feature is enabled -- that is an explicit request \
         for this backend. Skipping here would report `ok` for a test that ran nothing.",
    );
}

#[test]
fn f32_to_i64_matches_cpu_including_truncation() {
    let gpu = Wgpu::new(0);
    require_wgpu(&gpu);
    let source = [1.0f32, 2.7, -1.5, 0.0];
    let got = gpu
        .tensor(source)
        .unwrap()
        .to_dtype::<i64>()
        .unwrap()
        .to_vec1::<i64>()
        .unwrap();
    let want = Cpu
        .tensor(source)
        .unwrap()
        .to_dtype::<i64>()
        .unwrap()
        .to_vec1::<i64>()
        .unwrap();
    assert_eq!(got, want, "f32->i64 must agree with CPU");
    assert_eq!(got, vec![1, 2, -1, 0]);
}

#[test]
fn f32_to_u8_and_u32_match_cpu() {
    let gpu = Wgpu::new(0);
    require_wgpu(&gpu);
    let source = [0.0f32, 1.0, 2.7, 200.0];
    let got_u8 = gpu
        .tensor(source)
        .unwrap()
        .to_dtype::<u8>()
        .unwrap()
        .to_vec1::<u8>()
        .unwrap();
    let want_u8 = Cpu
        .tensor(source)
        .unwrap()
        .to_dtype::<u8>()
        .unwrap()
        .to_vec1::<u8>()
        .unwrap();
    assert_eq!(got_u8, want_u8, "f32->u8 must agree with CPU");
    assert_eq!(got_u8, vec![0u8, 1, 2, 200]);
    let got_u32 = gpu
        .tensor(source)
        .unwrap()
        .to_dtype::<u32>()
        .unwrap()
        .to_vec1::<u32>()
        .unwrap();
    let want_u32 = Cpu
        .tensor(source)
        .unwrap()
        .to_dtype::<u32>()
        .unwrap()
        .to_vec1::<u32>()
        .unwrap();
    assert_eq!(got_u32, want_u32, "f32->u32 must agree with CPU");
    assert_eq!(got_u32, vec![0u32, 1, 2, 200]);
}

#[test]
fn f32_to_f32_identity_round_trips() {
    let gpu = Wgpu::new(0);
    require_wgpu(&gpu);
    let source = [1.5f32, -2.25, 0.0, 3.125];
    let got = gpu
        .tensor(source)
        .unwrap()
        .to_dtype::<f32>()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(got, source);
}

#[test]
fn f64_target_is_refused_by_name() {
    let gpu = Wgpu::new(0);
    require_wgpu(&gpu);
    // Admission cannot see the target tag (it is an attribute), so this must
    // fail inside the executor — the refusal names the operation.
    let err = gpu
        .tensor([1.0f32, 2.0])
        .unwrap()
        .to_dtype::<f64>()
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("to_dtype"),
        "f64 target refusal must name to_dtype, got: {err}"
    );
}

#[test]
fn bool_and_q8_0_targets_are_refused_by_name() {
    let gpu = Wgpu::new(0);
    require_wgpu(&gpu);
    // `bool` has no conversion semantics on any backend (CPU refuses it too);
    // `Q8_0` is block-packed and entered through `quantize`, not a cast.
    for name in ["bool", "q8_0"] {
        let err = if name == "bool" {
            gpu.tensor([1.0f32, 0.0])
                .unwrap()
                .to_dtype::<bool>()
                .unwrap_err()
                .to_string()
        } else {
            gpu.tensor([0.5f32; 64])
                .unwrap()
                .to_dtype::<incin_core::tensor::dtype::Q8_0>()
                .unwrap_err()
                .to_string()
        };
        assert!(
            err.contains("to_dtype"),
            "{name} target refusal must name to_dtype, got: {err}"
        );
    }
}

#[test]
fn non_f32_sources_stay_refused_at_admission() {
    let gpu = Wgpu::new(0);
    require_wgpu(&gpu);
    // The capability row (`broadcast`, `F32_ONLY`) admits `f32` inputs only:
    // an integer source never reaches the executor's wider belt-and-suspenders
    // set. This pins the advertised scope, not a kernel gap.
    let err = gpu
        .tensor([1i64, 2])
        .unwrap()
        .to_dtype::<f32>()
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("to_dtype") || err.contains("Unsupported"),
        "i64 source must be refused, got: {err}"
    );
}

#[test]
fn to_dtype_is_advertised_for_f32() {
    let rule = incin_backends::capability::WGPU_CAPABILITIES
        .iter()
        .find(|rule| {
            rule.operation == OperationKind::ToDType
                && rule
                    .dtypes
                    .contains(&incin_core::tensor::dtype::DTypeId::F32.descriptor())
        });
    assert!(
        rule.is_some(),
        "WGPU_CAPABILITIES must advertise ToDType over f32"
    );
}
