#![cfg(feature = "cpu")]

use std::ops::Deref;
use std::sync::Arc;

use incin_backends::cpu::{CpuBackendImpl, CpuBuffer, CpuStorage};
#[cfg(any(feature = "wgpu", feature = "cuda", feature = "metal"))]
use incin_core::backend_authoring::Backend;
use incin_core::exec::{Alignment, LayoutClass, TensorMeta};
use incin_core::prelude::{DTypeId, DeviceId};
#[cfg(feature = "wgpu")]
use incin_core::__backend_compat::legacy::TensorOps;

fn assert_metadata_storage<T>()
where
    T: Deref<Target = TensorMeta>,
{
}

#[test]
fn every_enabled_backend_storage_uses_tensor_meta() {
    assert_metadata_storage::<
        <CpuBackendImpl as incin_core::backend_authoring::StorageBackend>::Storage<f32>,
    >();

    #[cfg(feature = "wgpu")]
    assert_metadata_storage::<<incin_backends::wgpu::WgpuBackendImpl as incin_core::backend_authoring::StorageBackend>::Storage<f32>>();

    #[cfg(feature = "cuda")]
    assert_metadata_storage::<<incin_backends::cuda::CudaBackendImpl as incin_core::backend_authoring::StorageBackend>::Storage<f32>>();
}

#[test]
fn contiguous_transpose_and_broadcast_have_one_checked_metadata_source() {
    let base = CpuStorage::try_from_contiguous(CpuBuffer::F32(vec![0.0; 6]), vec![2, 3]).unwrap();
    assert_eq!(&*base.shape, &[2, 3]);
    assert_eq!(&*base.strides, &[3, 1]);
    assert_eq!(base.offset_elements, 0);
    assert_eq!(base.dtype, DTypeId::F32.descriptor());
    assert_eq!(base.device, DeviceId::cpu());
    assert_eq!(base.layout, LayoutClass::Contiguous);
    assert!(base.alignment.supports(Alignment::of::<f32>().bytes()));
    assert!(std::ptr::eq(base.metadata(), &*base));

    let transposed = base.transpose(0, 1).unwrap();
    assert_eq!(&*transposed.shape, &[3, 2]);
    assert_eq!(&*transposed.strides, &[1, 3]);
    assert_eq!(transposed.layout, LayoutClass::Strided);

    let row = CpuStorage::try_from_contiguous(CpuBuffer::F32(vec![0.0; 3]), vec![1, 3]).unwrap();
    let broadcast = row.broadcast_as(&[4, 3]).unwrap();
    assert_eq!(&*broadcast.shape, &[4, 3]);
    assert_eq!(&*broadcast.strides, &[0, 1]);
    assert_eq!(broadcast.layout, LayoutClass::Strided);
}

#[test]
fn nonzero_view_offset_weakens_but_never_strengthens_alignment() {
    let base_alignment = Alignment::new(16).unwrap();
    let base = TensorMeta::try_new(
        [8].as_slice().into(),
        [1].as_slice().into(),
        0,
        DTypeId::F32.descriptor(),
        DeviceId::cpu(),
        base_alignment,
        8,
    )
    .unwrap();
    assert_eq!(base.alignment.bytes(), 16);

    let view = TensorMeta::try_new(
        [2].as_slice().into(),
        [1].as_slice().into(),
        1,
        DTypeId::F32.descriptor(),
        DeviceId::cpu(),
        base_alignment,
        8,
    )
    .unwrap();
    assert_eq!(view.offset_elements, 1);
    assert_eq!(view.alignment.bytes(), 4);
    assert!(!view.alignment.supports(16));
    assert!(view.alignment.supports(4));

    let storage = CpuStorage::try_from_contiguous(CpuBuffer::F32(vec![0.0; 8]), vec![8]).unwrap();
    let narrowed = storage.narrow(0, 3, 2).unwrap();
    assert_eq!(narrowed.offset_elements, 3);
    assert_eq!(&*narrowed.shape, &[2]);
}

#[test]
fn empty_views_are_valid_but_out_of_bounds_views_are_rejected() {
    let empty = CpuStorage::try_from_contiguous(CpuBuffer::F32(Vec::new()), vec![0, 3]).unwrap();
    assert_eq!(&*empty.shape, &[0, 3]);
    assert_eq!(empty.offset_elements, 0);
    assert_eq!(empty.layout, LayoutClass::Contiguous);

    let error =
        CpuStorage::try_from_parts(Arc::new(CpuBuffer::F32(vec![0.0; 4])), vec![3], vec![2], 1)
            .unwrap_err();
    let rendered = error.to_string();
    assert!(rendered.contains("out of bounds"), "{rendered}");
    assert!(rendered.contains("capacity 4"), "{rendered}");
}

#[cfg(feature = "wgpu")]
#[test]
fn wgpu_materialized_views_report_contiguous_zero_offset_metadata() {
    type WgpuB =
        incin_backends::wgpu::WgpuBackendImpl<incin_core::prelude::WgpuN<incin_core::typenum::U0>>;

    let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let storage = WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(&values),
        &[2, 3],
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .unwrap();
    let transposed = WgpuB::transpose::<f32>(&storage, 0, 1).unwrap();
    let metadata: &TensorMeta = &transposed;
    assert_eq!(&*metadata.shape, &[3, 2]);
    assert_eq!(&*metadata.strides, &[2, 1]);
    assert_eq!(metadata.offset_elements, 0);
    assert_eq!(metadata.layout, LayoutClass::Contiguous);
    assert_eq!(metadata.dtype, DTypeId::F32.descriptor());
    assert_eq!(metadata.device, DeviceId::wgpu(0));
}

/// CUDA metadata reports the alignment the device allocator actually provides.
///
/// EXE-004 originally recorded `Alignment::BYTE` here, because `CudaSlice<u8>`
/// is the only thing the Rust type system knows about a CUDA allocation and one
/// byte is all it proves. It is a true claim and a useless one: every CUDA
/// tensor would answer "unaligned" to a kernel choosing between a scalar and a
/// vector load. The first run on real hardware settled it — the driver returns
/// 256-byte-aligned addresses, as its own documentation promises — so the claim
/// now matches the allocator, and a view offset still weakens it.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device and driver"]
fn cuda_metadata_reports_the_measured_device_allocation_alignment() {
    type CudaB =
        incin_backends::cuda::CudaBackendImpl<incin_core::prelude::CudaN<incin_core::typenum::U0>>;

    let values = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let storage = CudaB::from_bytes::<f32>(
        bytemuck::cast_slice(&values),
        &[8],
        DTypeId::F32.descriptor(),
        &DeviceId::cuda(0),
    )
    .unwrap();
    let metadata: &TensorMeta = &storage;
    assert_eq!(metadata.device, DeviceId::cuda(0));
    assert!(
        metadata.alignment.supports(256),
        "CUDA storage reported {:?}",
        metadata.alignment
    );

    // One f32 into the allocation is four-byte aligned and nothing more, which
    // is the property that makes this a guarantee rather than a preference.
    let view = TensorMeta::try_new(
        [4].as_slice().into(),
        [1].as_slice().into(),
        1,
        DTypeId::F32.descriptor(),
        DeviceId::cuda(0),
        Alignment::new(256).unwrap(),
        8,
    )
    .unwrap();
    assert_eq!(view.alignment.bytes(), 4);
}

#[test]
fn invalid_alignment_rank_and_arithmetic_overflow_are_rejected() {
    assert!(Alignment::new(0).is_err());
    assert!(Alignment::new(3).is_err());

    let rank_error = CpuStorage::try_from_parts(
        Arc::new(CpuBuffer::F32(vec![0.0; 4])),
        vec![2, 2],
        vec![1],
        0,
    )
    .unwrap_err()
    .to_string();
    assert!(rank_error.contains("rank"), "{rank_error}");

    let overflow = TensorMeta::try_new(
        [2].as_slice().into(),
        [usize::MAX].as_slice().into(),
        1,
        DTypeId::F32.descriptor(),
        DeviceId::cpu(),
        Alignment::of::<f32>(),
        usize::MAX,
    )
    .unwrap_err()
    .to_string();
    assert!(overflow.contains("overflow"), "{overflow}");
}
