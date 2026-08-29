//! `transpose`, `flatten`, `squeeze` and `unsqueeze` on real WGPU hardware.
//!
//! These four are views: every one of them leaves the elements alone and
//! changes only how the shape describes them. That makes the interesting
//! assertion not "does it return something" but "are the *same numbers* still
//! there, in the order the new shape claims".
//!
//! `transpose` is the exception that has to move data, and it is also the one
//! that was already implemented and simply never registered, so it gets the
//! closest look: a rectangular (not square) input, because a square one hides
//! a row/column swap.
#![cfg(feature = "wgpu")]

use incin_backends::wgpu::WgpuBackendImpl;
use incin_core::backend_authoring::{HostInterop, HostReadback, StorageBackend, op};
use incin_core::exec::catalog::{AxisAttributes, FlattenAttributes, TransposeAttributes};
use incin_core::exec::{ExecutionContext, TensorHandle};
use incin_core::prelude::{DTypeId, DeviceId, WgpuN};
use incin_core::typenum::U0;

type TestBackend = WgpuBackendImpl<WgpuN<U0>>;
type TestStorage = <TestBackend as StorageBackend>::Storage<f32>;

/// 2x3, so a transpose is observable. Values are distinct so any reordering
/// mistake shows up as a different vector rather than a coincidence.
const VALUES: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

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

fn run<O>(input: &TestStorage, attributes: O::Attributes) -> TestStorage
where
    O: incin_core::exec::CanonicalOperation,
    TestBackend: incin_core::backend_authoring::Execute<O, Output = TestStorage>,
{
    let context = ExecutionContext::new(TestBackend::default());
    let inputs = [TensorHandle::from_storage::<TestBackend, f32, _>(input)];
    incin_core::exec::dispatch::execute::<O, _>(&context, attributes, &inputs)
        .expect("an advertised operation must execute")
}

/// The kernel that existed but was never advertised. A rectangular input, so
/// a swap that did nothing would be caught.
#[test]
fn transpose_reorders_a_rectangular_tensor() {
    if !has_wgpu() {
        return;
    }
    let input = upload(&VALUES, &[2, 3]);
    let out = run::<op::TransposeExact>(
        &input,
        TransposeAttributes {
            first: 0,
            second: 1,
        },
    );

    // [[1,2,3],[4,5,6]] transposed is [[1,4],[2,5],[3,6]].
    assert_eq!(read(&out), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

/// Transposing twice is the identity, which also exercises the path the tape
/// entry uses for its backward.
#[test]
fn transpose_is_its_own_inverse() {
    if !has_wgpu() {
        return;
    }
    let input = upload(&VALUES, &[2, 3]);
    let swap = || TransposeAttributes {
        first: 0,
        second: 1,
    };
    let once = run::<op::TransposeExact>(&input, swap());
    let twice = run::<op::TransposeExact>(&once, swap());

    assert_eq!(read(&twice), read(&input));
}

#[test]
fn flatten_collapses_an_axis_range_without_moving_data() {
    if !has_wgpu() {
        return;
    }
    let input = upload(&VALUES, &[1, 2, 3]);
    let out = run::<op::FlattenExact>(
        &input,
        FlattenAttributes {
            start_axis: 1,
            end_axis: 2,
        },
    );

    assert_eq!(
        read(&out),
        VALUES.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
        "a flatten is a view: the elements must be untouched"
    );
}

#[test]
fn squeeze_drops_a_unit_axis_without_moving_data() {
    if !has_wgpu() {
        return;
    }
    let input = upload(&VALUES, &[1, 2, 3]);
    let out = run::<op::SqueezeExact>(&input, AxisAttributes { axis: 0 });

    assert_eq!(
        read(&out),
        VALUES.iter().map(|v| f64::from(*v)).collect::<Vec<_>>()
    );
}

#[test]
fn unsqueeze_then_squeeze_round_trips() {
    if !has_wgpu() {
        return;
    }
    let input = upload(&VALUES, &[2, 3]);
    let widened = run::<op::UnsqueezeExact>(&input, AxisAttributes { axis: 0 });
    let narrowed = run::<op::SqueezeExact>(&widened, AxisAttributes { axis: 0 });

    assert_eq!(
        read(&narrowed),
        read(&input),
        "inserting and removing the same unit axis returns the original"
    );
}

/// `squeeze` on an axis that is not extent 1 has to refuse. Quietly keeping
/// the axis would hand back a tensor of a different rank than the caller
/// asked for, which every downstream shape check would then trust.
#[test]
fn squeeze_refuses_a_non_unit_axis() {
    if !has_wgpu() {
        return;
    }
    let input = upload(&VALUES, &[2, 3]);
    let context = ExecutionContext::new(TestBackend::default());
    let inputs = [TensorHandle::from_storage::<TestBackend, f32, _>(&input)];

    let outcome = incin_core::exec::dispatch::execute::<op::SqueezeExact, _>(
        &context,
        AxisAttributes { axis: 0 },
        &inputs,
    );

    assert!(
        outcome.is_err(),
        "squeezing an axis of extent 2 must be refused, not silently ignored"
    );
}
