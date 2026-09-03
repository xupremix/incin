//! `maximum`, `minimum` and `abs_diff` on real WGPU hardware.
//!
//! These three are the first members of WGPU's `native_tensor` group, and that
//! group is declared `training = true`. A capability row claiming training is a
//! claim that the operation records on the tape, and `shaders/binary.wgsl` has
//! carried fused modes for all three since it was written: reaching for the
//! fused mode alone would have produced correct numbers and no gradient, which
//! reads as working until a model silently stops learning.
//!
//! So each case checks two separate things: the forward against the CPU
//! reference's own values, and that the operation left something on the tape a
//! backward pass can walk.
//!
//! The tie cases are deliberate. `cpu::canonical`'s `Execute<op::Maximum>`
//! masks on `a > b` and selects `where(mask, lhs, rhs)`, so `maximum(3, 3)`
//! resolves to the *right* operand and hands it the whole cotangent. A mask
//! facing `>=` would agree on every value in this file and disagree on every
//! tie, so the tie is the only input that distinguishes them.
#![cfg(feature = "wgpu")]

use incin_backends::wgpu::WgpuBackendImpl;
use incin_core::backend_authoring::{HostInterop, HostReadback, StorageBackend, op};
use incin_core::exec::catalog::NoAttributes;
use incin_core::exec::{ExecutionContext, TensorHandle};
use incin_core::prelude::{DTypeId, DeviceId, WgpuN};
use incin_core::typenum::U0;

type TestBackend = WgpuBackendImpl<WgpuN<U0>>;
type TestStorage = <TestBackend as StorageBackend>::Storage<f32>;

/// Operands chosen so every branch of a selection is exercised: left wins,
/// right wins, a tie, a negative pair, and an equal pair away from zero.
const LHS: [f32; 6] = [1.0, 5.0, 3.0, 3.0, -2.0, 0.5];
const RHS: [f32; 6] = [4.0, 2.0, 3.0, 3.0, -7.0, 0.5];

/// Aborts unless a WGPU adapter is present.
///
/// Replaces a `has_wgpu() -> bool` predicate that callers used to skip with an
/// early `return`. That reports `ok` for a test that ran nothing, so the job
/// named "WGPU Software Adapter Tests" stayed green whether or not an adapter
/// existed -- the same defect that let three CUDA bugs survive behind suites
/// that launched no kernel.
///
/// Failing is right here because these suites are `#![cfg(feature = "wgpu")]`:
/// compiling them at all is an explicit request for the backend, and both CI
/// jobs that enable it install a software adapter first.
///
/// # Panics
///
/// If no WGPU adapter can be reached.
fn require_wgpu() {
    assert!(
        <TestBackend as HostInterop>::from_bytes::<f32>(
            &[0u8; 4],
            &[1],
            DTypeId::F32.descriptor(),
            &DeviceId::wgpu(0),
        )
        .is_ok(),
        "no WGPU adapter, but the `wgpu` feature is enabled -- that is an explicit request for this backend. Skipping here would report `ok` for a test that ran nothing."
    );
}

fn upload(values: &[f32]) -> TestStorage {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    <TestBackend as HostInterop>::from_bytes::<f32>(
        &bytes,
        &[values.len()],
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .expect("uploading six f32 values must succeed")
}

fn read(storage: &TestStorage) -> Vec<f64> {
    <TestBackend as HostReadback>::float_to_vec1::<f32>(storage)
        .expect("reading a contiguous f32 buffer back must succeed")
}

/// Runs one two-operand catalog operation and returns its result and the number
/// of tape entries it added.
fn run<O>(lhs: &TestStorage, rhs: &TestStorage) -> (TestStorage, usize)
where
    O: incin_core::exec::CanonicalOperation<Attributes = NoAttributes>,
    TestBackend: incin_core::backend_authoring::Execute<O, Output = TestStorage>,
{
    let context = ExecutionContext::new(TestBackend::default());
    let inputs = [
        TensorHandle::from_storage::<TestBackend, f32, _>(lhs),
        TensorHandle::from_storage::<TestBackend, f32, _>(rhs),
    ];
    let before = incin_backends::wgpu::tape_depth();
    let out = incin_core::exec::dispatch::execute::<O, _>(&context, NoAttributes, &inputs)
        .expect("an advertised operation must execute");
    (out, incin_backends::wgpu::tape_depth() - before)
}

#[test]
fn maximum_matches_the_cpu_reference_and_records_a_gradient() {
    require_wgpu();
    let (lhs, rhs) = (upload(&LHS), upload(&RHS));
    let (out, recorded) = run::<op::Maximum>(&lhs, &rhs);
    assert_eq!(read(&out), vec![4.0, 5.0, 3.0, 3.0, -2.0, 0.5]);
    assert_eq!(
        recorded, 1,
        "maximum advertises `training = true`, so it must leave exactly one \
         tape entry: the fused forward plus the mask-splitting backward. A bare \
         fused dispatch would return these same numbers and record none, and a \
         composed forward would record more than one."
    );
}

#[test]
fn minimum_matches_the_cpu_reference_and_records_a_gradient() {
    require_wgpu();
    let (lhs, rhs) = (upload(&LHS), upload(&RHS));
    let (out, recorded) = run::<op::Minimum>(&lhs, &rhs);
    assert_eq!(read(&out), vec![1.0, 2.0, 3.0, 3.0, -7.0, 0.5]);
    assert_eq!(recorded, 1, "minimum records one entry, as maximum does");
}

#[test]
fn abs_diff_matches_the_cpu_reference_and_records_a_gradient() {
    require_wgpu();
    let (lhs, rhs) = (upload(&LHS), upload(&RHS));
    let (out, recorded) = run::<op::AbsDiff>(&lhs, &rhs);
    assert_eq!(read(&out), vec![3.0, 3.0, 0.0, 0.0, 5.0, 0.0]);
    // `abs_diff` composes `sub` then `abs`, so it records one entry per step.
    assert!(
        recorded >= 2,
        "abs_diff composes two tape-tracked primitives, so it records both"
    );
}
