//! `rms_norm` on real WGPU hardware, against an independent host reference.
//!
//! WGPU advertises this as `Composed` with `training = true`. It is answered
//! by rewriting into `mul`, `mean_keepdim`, `add_scalar`, `sqrt` and `div`,
//! the same chain CPU and CUDA use, so the three should agree on the same
//! input rather than each being separately plausible.
//!
//! The epsilon guard is the subtle part, and the two things worth pinning about
//! it are pinned by two different tests, which is worth stating because the
//! obvious guess is wrong. Removing the guard entirely is caught by the
//! all-zero row, where `sqrt(0)` becomes a divisor. Moving the guard to
//! *after* the square root is not caught there -- `sqrt(0) + eps` is still a
//! fine divisor -- it is caught by the reference comparison, because
//! `sqrt(mean + eps)` and `sqrt(mean) + eps` differ on ordinary input. Both
//! mutations were run against this file to confirm which test catches which.
#![cfg(feature = "wgpu")]

use incin_backends::wgpu::WgpuBackendImpl;
use incin_core::backend_authoring::{HostInterop, HostReadback, StorageBackend, op};
use incin_core::exec::catalog::EpsilonAttributes;
use incin_core::exec::{ExecutionContext, TensorHandle};
use incin_core::prelude::{DTypeId, DeviceId, WgpuN};
use incin_core::typenum::U0;

type TestBackend = WgpuBackendImpl<WgpuN<U0>>;
type TestStorage = <TestBackend as StorageBackend>::Storage<f32>;

const EPSILON: f64 = 1e-5;

fn has_wgpu() -> bool {
    <TestBackend as HostInterop>::from_bytes::<f32>(
        &[0u8; 4],
        &[1],
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .is_ok()
}

fn upload(values: &[f32], shape: &[usize]) -> TestStorage {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    <TestBackend as HostInterop>::from_bytes::<f32>(
        &bytes,
        shape,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .expect("uploading the operand must succeed")
}

fn read(storage: &TestStorage) -> Vec<f64> {
    <TestBackend as HostReadback>::float_to_vec1::<f32>(storage)
        .expect("reading a contiguous f32 buffer back must succeed")
}

fn rms_norm(values: &[f32], shape: &[usize], weight: &[f32]) -> (Vec<f64>, usize) {
    let input = upload(values, shape);
    let weight_storage = upload(weight, &[weight.len()]);
    let context = ExecutionContext::new(TestBackend::default());
    let inputs = [
        TensorHandle::from_storage::<TestBackend, f32, _>(&input),
        TensorHandle::from_storage::<TestBackend, f32, _>(&weight_storage),
    ];
    let before = incin_backends::wgpu::tape_depth();
    let out = incin_core::exec::dispatch::execute::<op::RmsNorm, _>(
        &context,
        EpsilonAttributes { epsilon: EPSILON },
        &inputs,
    )
    .expect("an advertised operation must execute");
    (read(&out), incin_backends::wgpu::tape_depth() - before)
}

/// `x / sqrt(mean(x^2) + eps) * weight`, over the last axis.
fn reference(values: &[f32], rows: usize, cols: usize, weight: &[f32]) -> Vec<f64> {
    let mut out = vec![0.0f64; values.len()];
    for row in 0..rows {
        let slice = &values[row * cols..(row + 1) * cols];
        let mean_square: f64 = slice
            .iter()
            .map(|v| f64::from(*v) * f64::from(*v))
            .sum::<f64>()
            / cols as f64;
        let scale = (mean_square + EPSILON).sqrt();
        for (column, value) in slice.iter().enumerate() {
            out[row * cols + column] = (f64::from(*value) / scale) * f64::from(weight[column]);
        }
    }
    out
}

#[test]
fn rms_norm_matches_the_reference_and_records_a_gradient() {
    if !has_wgpu() {
        return;
    }
    const VALUES: [f32; 8] = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5, -2.0, 3.0];
    const WEIGHT: [f32; 4] = [1.0, 0.5, 2.0, 1.5];

    let (got, recorded) = rms_norm(&VALUES, &[2, 4], &WEIGHT);
    let want = reference(&VALUES, 2, 4, &WEIGHT);

    for (index, (got, want)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "element {index}: got {got}, reference {want}"
        );
    }

    assert!(
        recorded >= 2,
        "rms_norm advertises `training = true` and is composed from six \
         tape-tracked primitives, so it must leave entries behind. A forward \
         that recorded none would return these same numbers and silently \
         drop the gradient."
    );
}

/// The weight is applied per feature, so changing one component must change
/// exactly one column. A kernel that ignored the weight, or broadcast it the
/// wrong way, passes a whole-tensor comparison against a uniform weight and
/// fails this.
#[test]
fn the_weight_is_applied_per_feature() {
    if !has_wgpu() {
        return;
    }
    const VALUES: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let uniform = rms_norm(&VALUES, &[1, 4], &[1.0, 1.0, 1.0, 1.0]).0;
    let scaled = rms_norm(&VALUES, &[1, 4], &[1.0, 1.0, 3.0, 1.0]).0;

    assert!((scaled[0] - uniform[0]).abs() < 1e-6, "column 0 unchanged");
    assert!((scaled[1] - uniform[1]).abs() < 1e-6, "column 1 unchanged");
    assert!((scaled[3] - uniform[3]).abs() < 1e-6, "column 3 unchanged");
    assert!(
        (scaled[2] - uniform[2] * 3.0).abs() < 1e-6,
        "column 2 must scale by exactly its weight: {} vs {}",
        scaled[2],
        uniform[2] * 3.0
    );
}

/// Each row is normalised independently, so a row of larger magnitude must not
/// pull on a row beside it. Reducing the wrong axis breaks this first.
#[test]
fn rows_are_normalised_independently() {
    if !has_wgpu() {
        return;
    }
    const SMALL: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    const WEIGHT: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

    let alone = rms_norm(&SMALL, &[1, 4], &WEIGHT).0;

    // The same row, now sitting beside a much larger one.
    let together_values = [1.0, 2.0, 3.0, 4.0, 1000.0, 2000.0, 3000.0, 4000.0];
    let together = rms_norm(&together_values, &[2, 4], &WEIGHT).0;

    for column in 0..4 {
        assert!(
            (together[column] - alone[column]).abs() < 1e-6,
            "column {column} changed when a larger row was added: {} vs {}",
            together[column],
            alone[column]
        );
    }
}

/// Why the guard exists at all. Without it an all-zero row has a mean square of
/// zero, so the divisor is `sqrt(0)`, and every element comes back NaN.
///
/// This test does *not* pin where the epsilon goes: `sqrt(0) + eps` is also a
/// usable divisor, so a version that added it after the root passes here. The
/// ordering is pinned by the reference comparison above instead.
#[test]
fn an_all_zero_row_stays_finite() {
    if !has_wgpu() {
        return;
    }
    const ZEROS: [f32; 4] = [0.0, 0.0, 0.0, 0.0];
    const WEIGHT: [f32; 4] = [1.0, 1.0, 1.0, 1.0];

    let (got, _) = rms_norm(&ZEROS, &[1, 4], &WEIGHT);

    assert!(
        got.iter().all(|value| value.is_finite()),
        "the epsilon guard must keep an all-zero row finite, got {got:?}"
    );
    assert!(
        got.iter().all(|value| *value == 0.0),
        "and zero divided by a positive scale is still zero, got {got:?}"
    );
}
