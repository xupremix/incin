//! Device-to-backend selection for the unified `IncinBackend` facade.

use incin_core::prelude::{Backend, DType, Device, HostInterop, StorageBackend, VariableBackend};
#[cfg(test)]
use incin_core::prelude::{Cpu, Dyn, ShapeBuf};

macro_rules! impl_transfer {
    ($source:ty) => {
        impl<D: Device, NewD: Device> incin_core::prelude::TransferTo<NewD> for $source
        where
            crate::target::Native: crate::target::EngineOn<NewD>,
        {
            type Output = crate::IncinBackend<NewD>;

            fn transfer_storage<K: DType>(
                storage: &Self::Storage<K>,
                dtype: &K::Field,
                device: &NewD::Field,
            ) -> incin_core::prelude::Result<
                <Self::Output as incin_core::backend_authoring::StorageBackend>::Storage<K>,
            >
            where
                Self::Output: incin_core::prelude::SupportsDType<K>,
            {
                use incin_core::prelude::{Error, SupportsDType};
                let expected_descriptor = K::descriptor(dtype);
                let source_dtype = Self::storage_dtype::<K>(storage).ok_or(
                    Error::UnsupportedBackendOperation {
                        op: "transfer_storage_metadata",
                        backend: core::any::type_name::<Self>(),
                    },
                )?;
                if source_dtype != expected_descriptor {
                    return Err(Error::DTypeStorageMismatch {
                        expected: expected_descriptor,
                        got: source_dtype,
                    });
                }
                Self::storage_device::<K>(storage).ok_or(Error::UnsupportedBackendOperation {
                    op: "transfer_storage_metadata",
                    backend: core::any::type_name::<Self>(),
                })?;
                let destination = NewD::to_incin(device)?;
                let dtype_descriptor =
                    <Self::Output as SupportsDType<K>>::resolve_dtype(dtype, &destination)?;
                let shape = Self::shape::<K>(storage);
                let bytes = Self::to_bytes::<K>(storage)?;
                <Self::Output as HostInterop>::from_bytes::<K>(
                    &bytes,
                    &shape,
                    dtype_descriptor,
                    &destination,
                )
            }

            fn transfer_var<K: DType>(
                variable: &Self::RawVar,
                dtype: &K::Field,
                device: &NewD::Field,
            ) -> incin_core::prelude::Result<<Self::Output as VariableBackend>::RawVar>
            where
                Self::Output: incin_core::prelude::SupportsDType<K>,
            {
                use incin_core::prelude::SupportsDType;
                let source = <Self as VariableBackend>::var_as_tensor::<K>(variable)?;
                let expected_descriptor = K::descriptor(dtype);
                if let Some(got) = Self::storage_dtype::<K>(&source)
                    && got != expected_descriptor
                {
                    return Err(incin_core::prelude::Error::DTypeStorageMismatch {
                        expected: expected_descriptor,
                        got,
                    });
                }
                let destination = NewD::to_incin(device)?;
                let dtype_descriptor =
                    <Self::Output as SupportsDType<K>>::resolve_dtype(dtype, &destination)?;
                let shape = Self::shape::<K>(&source);
                let bytes = Self::to_bytes::<K>(&source)?;
                let storage = <Self::Output as HostInterop>::from_bytes::<K>(
                    &bytes,
                    &shape,
                    dtype_descriptor,
                    &destination,
                )?;
                <Self::Output as VariableBackend>::var_from_tensor(&storage)
            }
        }
    };
}

#[cfg(feature = "cpu")]
impl_transfer!(crate::cpu::CpuBackendImpl<D>);
#[cfg(feature = "wgpu")]
impl_transfer!(crate::wgpu::WgpuBackendImpl<D>);
#[cfg(feature = "cuda")]
impl_transfer!(crate::cuda::CudaBackendImpl<D>);
#[cfg(feature = "metal")]
impl_transfer!(crate::metal::MetalBackendImpl<D>);
impl_transfer!(crate::dispatch::DispatchBackend<D>);

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "cpu")]
    use incin_core::exec::DescriptorError;
    #[cfg(all(feature = "cpu", not(feature = "wgpu")))]
    use incin_core::prelude::BackendError;
    #[cfg(feature = "cpu")]
    use incin_core::prelude::{
        DTypeId, DeviceId, Error, Grad, LayerNorm, Linear, OperationKind, RequiresGrad, Tensor,
        ToDevice,
    };
    #[cfg(feature = "cpu")]
    type Linear23 = incin_core::shapes::shape::DimCons<
        incin_core::typenum::U2,
        incin_core::shapes::shape::DimCons<incin_core::typenum::U3, incin_core::shapes::shape::Nil>,
    >;

    fn assert_backend<B: Backend>() {}

    #[cfg(feature = "cpu")]
    #[test]
    fn selects_cpu_backend() {
        assert_backend::<crate::IncinBackend<Cpu>>();
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn selects_wgpu_backend() {
        assert_backend::<crate::IncinBackend<incin_core::prelude::Wgpu>>();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn selects_cuda_backend() {
        assert_backend::<crate::IncinBackend<incin_core::prelude::Cuda>>();
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn runtime_dispatch_selects_cpu_and_preserves_metadata() {
        type B = crate::IncinBackend<Dyn>;
        let tensor = Tensor::<Dyn, B, Dyn>::zeros(([2, 3], DTypeId::F64, DeviceId::cpu())).unwrap();
        assert_eq!(tensor.dims(), vec![2, 3]);
        assert_eq!(tensor.dtype(), DTypeId::F64.descriptor());
        assert_eq!(tensor.device().unwrap(), DeviceId::cpu());
        assert!(matches!(
            tensor.inner(),
            crate::dispatch::DispatchStorage::Cpu(_)
        ));
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn checked_storage_wrapping_rejects_metadata_mismatches() {
        type B = crate::IncinBackend<Dyn>;
        let storage =
            Tensor::<Dyn, B, Dyn>::zeros(([1], DTypeId::F32.descriptor(), DeviceId::cpu()))
                .unwrap()
                .into_inner();

        let dtype_error = Tensor::<Dyn, B, Dyn, Grad>::try_from_storage(
            storage.clone(),
            ShapeBuf::from_slice(&[1]),
            DTypeId::F64.descriptor(),
            DeviceId::cpu(),
            <Grad as RequiresGrad>::init(()),
        )
        .unwrap_err();
        assert!(matches!(
            dtype_error,
            Error::DTypeStorageMismatch {
                expected,
                got,
            } if expected == DTypeId::F64.descriptor() && got == DTypeId::F32.descriptor()
        ));

        let device_error = Tensor::<Dyn, B, Dyn, Grad>::try_from_storage(
            storage,
            ShapeBuf::from_slice(&[1]),
            DTypeId::F32.descriptor(),
            DeviceId::wgpu(0),
            <Grad as RequiresGrad>::init(()),
        )
        .unwrap_err();
        assert!(matches!(device_error, Error::DeviceStorageMismatch { .. }));
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn runtime_from_bytes_rejects_wrong_length() {
        type B = crate::IncinBackend<Dyn>;
        let error = Tensor::<Dyn, B, Dyn>::from_bytes(
            &[0; 3],
            ([1], DTypeId::F32.descriptor(), DeviceId::cpu()),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            Error::Descriptor(DescriptorError::PayloadByteLength {
                operation: OperationKind::TensorFromBytes,
                ..
            })
        ));
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn runtime_dispatch_reduces_and_transfers_through_host_bytes() {
        type B = crate::IncinBackend<Dyn>;
        let tensor =
            Tensor::<Dyn, B, Dyn>::ones(([2, 2], DTypeId::F32.descriptor(), DeviceId::cpu()))
                .unwrap();
        let reduced = tensor.clone().sum_all().unwrap();
        assert_eq!(reduced.to_scalar::<f32>().unwrap(), 4.0);

        let transferred =
            <Tensor<Dyn, B, Dyn> as incin_core::tensor::transfer::ToDevice<B, Dyn>>::to_device(
                tensor,
                &DeviceId::cpu(),
            )
            .unwrap();
        assert_eq!(transferred.dims(), vec![2, 2]);
        assert_eq!(transferred.dtype(), DTypeId::F32.descriptor());
        assert_eq!(transferred.device().unwrap(), DeviceId::cpu());
        assert_eq!(
            transferred.sum_all().unwrap().to_scalar::<f32>().unwrap(),
            4.0
        );
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn device_transfer_detaches_tracked_tensor_type() {
        type B = crate::IncinBackend<Dyn>;
        let tensor =
            Tensor::<Dyn, B, Dyn, Grad>::ones(([1], DTypeId::F32.descriptor(), DeviceId::cpu()))
                .unwrap();
        let transferred =
            <Tensor<Dyn, B, Dyn, Grad> as incin_core::tensor::transfer::ToDevice<B, Dyn>>::to_device(
                tensor,
                &DeviceId::cpu(),
            )
            .unwrap();
        fn assert_detached(_: &Tensor<Dyn, B, Dyn, incin_core::prelude::NoGrad>) {}
        assert_detached(&transferred);
    }

    #[cfg(all(feature = "cpu", not(feature = "wgpu")))]
    #[test]
    fn runtime_dispatch_reports_disabled_backend() {
        type B = crate::IncinBackend<Dyn>;
        let error =
            Tensor::<Dyn, B, Dyn>::zeros(([1], DTypeId::F32.descriptor(), DeviceId::wgpu(0)))
                .unwrap_err();
        assert!(matches!(
            error,
            Error::Backend(BackendError::Execution {
                operation: OperationKind::Zeros,
                ..
            })
        ));
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn static_cpu_transfer_rebuilds_dynamic_dispatch_storage() {
        type Source = crate::IncinBackend<Cpu>;
        type Target = crate::IncinBackend<Dyn>;
        let tensor = Tensor::<Dyn, Source>::from_slice(&[1.0f32, 2.0, 3.0], [3]).unwrap();
        let transferred =
            <Tensor<Dyn, Source> as incin_core::tensor::transfer::ToDevice<Source, Dyn>>::to_device(
                tensor,
                &DeviceId::cpu(),
            )
            .unwrap();
        fn assert_target(_: &Tensor<Dyn, Target>) {}
        assert_target(&transferred);
        assert!(matches!(
            transferred.inner(),
            crate::dispatch::DispatchStorage::Cpu(_)
        ));
        assert_eq!(transferred.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn dynamic_layer_builders_consume_dtype_device_and_optional_flags() {
        type B = crate::IncinBackend<Dyn>;
        let linear = Linear::<Dyn, B>::build((10, 20, DeviceId::cpu())).unwrap();
        assert_eq!(linear.weight.shape_dims(), vec![20, 10]);

        let biased = Linear::<Dyn, B, Dyn>::build((10, 20, DeviceId::cpu(), true)).unwrap();
        assert!(biased.bias.is_some());

        let norm = LayerNorm::<Dyn, B>::build((20, DeviceId::cpu(), 1e-5)).unwrap();
        assert_eq!(norm.weight.shape_dims(), vec![20]);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn derived_module_transfer_changes_its_backend_output_type() {
        type Source = crate::IncinBackend<Cpu>;
        type Target = crate::IncinBackend<Dyn>;
        let layer = Linear::<Linear23, Source>::build(()).unwrap();
        let transferred = ToDevice::<Source, Dyn>::to_device(layer, &DeviceId::cpu()).unwrap();
        fn assert_target(_: &Linear<Linear23, Target>) {}
        assert_target(&transferred);
        assert!(matches!(
            transferred.weight.as_tensor().unwrap().inner(),
            crate::dispatch::DispatchStorage::Cpu(_)
        ));
    }
}
