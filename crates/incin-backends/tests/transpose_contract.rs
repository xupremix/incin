//! The settled transpose contract, issue #113, checked across backends.
//!
//! `TransposeExact` materialises a fresh dense row-major result on every
//! backend that advertises it -- the property the public `transpose`'s
//! `RowMajor` claim rests on -- and `TransposeView` is advertised only where
//! the backend can serve a genuine view. CPU and WGPU are pinned against the
//! same constants so a divergence fails the same assertion on both.
#![cfg(all(feature = "cpu", feature = "wgpu"))]

use incin_backends::cpu::CpuBackendImpl;
use incin_backends::wgpu::WgpuBackendImpl;
use incin_core::backend_authoring::{Execute, HostInterop, HostReadback, StorageBackend, op};
use incin_core::exec::catalog::TransposeAttributes;
use incin_core::exec::meta::LayoutClass;
use incin_core::exec::policy::MathMode;
use incin_core::exec::{
    CanonicalOperation, Capabilities, CapabilityQuery, ExecutionContext, OperationIdentity,
    SupportLevel, TensorHandle,
};
use incin_core::prelude::{DTypeId, DeviceId, OperationKind, WgpuN};
use incin_core::typenum::U0;

type Cpu = CpuBackendImpl;
type Wgpu = WgpuBackendImpl<WgpuN<U0>>;
type CpuStorage = <Cpu as StorageBackend>::Storage<f32>;
type WgpuStorage = <Wgpu as StorageBackend>::Storage<f32>;

/// 2x3 with distinct values, so a reordering mistake surfaces as a different
/// vector rather than a coincidence.
const VALUES: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// [[1, 2, 3], [4, 5, 6]] transposed is [[1, 4], [2, 5], [3, 6]].
const TRANSPOSED: [f64; 6] = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0];
const INPUT_SHAPE: [usize; 2] = [2, 3];
const OUTPUT_SHAPE: [usize; 2] = [3, 2];
/// Row-major strides for shape [3, 2]: a materialised transpose.
const DENSE_STRIDES: [usize; 2] = [2, 1];
/// The input's strides [3, 1] with the pair swapped: a view over the input.
const VIEW_STRIDES: [usize; 2] = [1, 3];

const SWAP: TransposeAttributes = TransposeAttributes {
    first: 0,
    second: 1,
};

fn bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

/// Aborts unless a WGPU adapter is present, like every other WGPU suite:
/// enabling the feature is an explicit request for the backend, and a skip
/// would report `ok` for a test that ran nothing.
fn require_wgpu() {
    assert!(
        <Wgpu as HostInterop>::from_bytes::<f32>(
            &[0u8; 4],
            &[1],
            DTypeId::F32.descriptor(),
            &DeviceId::wgpu(0),
        )
        .is_ok(),
        "no WGPU adapter, but the `wgpu` feature is enabled -- that is an explicit request for this backend"
    );
}

fn cpu_upload() -> CpuStorage {
    <Cpu as HostInterop>::from_bytes::<f32>(
        &bytes(&VALUES),
        &INPUT_SHAPE,
        DTypeId::F32.descriptor(),
        &DeviceId::cpu(),
    )
    .expect("uploading to CPU storage must succeed")
}

fn wgpu_upload() -> WgpuStorage {
    <Wgpu as HostInterop>::from_bytes::<f32>(
        &bytes(&VALUES),
        &INPUT_SHAPE,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .expect("uploading to WGPU storage must succeed")
}

fn cpu_execute<O>(input: &CpuStorage, attributes: O::Attributes) -> CpuStorage
where
    O: CanonicalOperation,
    Cpu: Execute<O, Output = CpuStorage>,
{
    let context = ExecutionContext::new(Cpu::default());
    let inputs = [TensorHandle::from_storage::<Cpu, f32, _>(input)];
    incin_core::exec::dispatch::execute::<O, _>(&context, attributes, &inputs)
        .expect("an advertised CPU operation must execute")
}

fn wgpu_execute<O>(input: &WgpuStorage, attributes: O::Attributes) -> WgpuStorage
where
    O: CanonicalOperation,
    Wgpu: Execute<O, Output = WgpuStorage>,
{
    require_wgpu();
    let context = ExecutionContext::new(Wgpu::default());
    let inputs = [TensorHandle::from_storage::<Wgpu, f32, _>(input)];
    incin_core::exec::dispatch::execute::<O, _>(&context, attributes, &inputs)
        .expect("an advertised WGPU operation must execute")
}

#[test]
fn cpu_transpose_exact_materialises_a_dense_row_major_result() {
    let input = cpu_upload();
    let out = cpu_execute::<op::TransposeExact>(&input, SWAP);
    let meta = <Cpu as StorageBackend>::metadata::<f32>(&out);
    assert_eq!(meta.shape().as_ref(), &OUTPUT_SHAPE);
    assert_eq!(
        meta.strides().as_ref(),
        &DENSE_STRIDES,
        "TransposeExact must copy into row-major strides; [1, 3] would mean it viewed again"
    );
    let values = <Cpu as HostReadback>::float_to_vec1::<f32>(&out)
        .expect("reading the materialised result back must succeed");
    assert_eq!(values, TRANSPOSED);
}

#[test]
fn cpu_transpose_view_permutes_strides_without_copying() {
    let input = cpu_upload();
    let out = cpu_execute::<op::TransposeView>(&input, SWAP);
    let meta = <Cpu as StorageBackend>::metadata::<f32>(&out);
    assert_eq!(meta.shape().as_ref(), &OUTPUT_SHAPE);
    assert_eq!(
        meta.strides().as_ref(),
        &VIEW_STRIDES,
        "TransposeView must permute the input's strides; [2, 1] would mean it copied"
    );
    let values = <Cpu as HostReadback>::float_to_vec1::<f32>(&out)
        .expect("reading the view back must succeed");
    assert_eq!(values, TRANSPOSED);
}

#[test]
fn wgpu_transpose_exact_matches_the_cpu_contract() {
    let input = wgpu_upload();
    let out = wgpu_execute::<op::TransposeExact>(&input, SWAP);
    let meta = <Wgpu as StorageBackend>::metadata::<f32>(&out);
    assert_eq!(meta.shape().as_ref(), &OUTPUT_SHAPE);
    assert_eq!(
        meta.strides().as_ref(),
        &DENSE_STRIDES,
        "the same operation on the same constants must produce the same memory order on every backend (issue #113)"
    );
    let values = <Wgpu as HostReadback>::float_to_vec1::<f32>(&out)
        .expect("reading the materialised result back must succeed");
    assert_eq!(values, TRANSPOSED);
}

#[test]
fn wgpu_advertises_transpose_exact_but_refuses_transpose_view() {
    require_wgpu();
    let backend = Wgpu::default();
    let base = |operation: OperationKind| CapabilityQuery {
        operation: OperationIdentity::Builtin(operation),
        dtype: DTypeId::F32.descriptor(),
        layout: LayoutClass::Contiguous,
        rank: 2,
        training: false,
        math_mode: MathMode::default(),
    };
    let exact = base(OperationKind::TransposeExact);
    assert!(
        !matches!(backend.support(&exact), SupportLevel::Unsupported(_)),
        "WGPU advertises TransposeExact; the parity test above executes it"
    );
    let view = base(OperationKind::TransposeView);
    assert!(
        matches!(backend.support(&view), SupportLevel::Unsupported(_)),
        "WGPU must refuse TransposeView -- its pointwise shaders address \
         linearly and would read a view's elements in the wrong order, so \
         the capability registry omits the row (issue #113)"
    );
}
