//! Device-to-backend selection for the unified `IncinBackend` facade.

#[cfg(feature = "cpu")]
use incin_core::prelude::Cpu;
use incin_core::prelude::{Backend, DType, Device, Dyn};

mod sealed {
    pub trait Sealed<T> {}
}

/// Maps a type-level device and float element type to its concrete backend.
///
/// `IncinBackend<T, D>` is a type alias through this trait, so it retains
/// the concrete backend storage and operation implementations without a
/// runtime-dispatch wrapper.
pub trait BackendFor<T: DType>: Device + sealed::Sealed<T> {
    /// Concrete backend for this device and float element type.
    type Backend: Backend<Device = Self, FloatElem = T, IntElem = i64>;
}

#[cfg(feature = "cpu")]
impl<T: DType> sealed::Sealed<T> for Cpu {}

#[cfg(feature = "cpu")]
impl<T: DType> BackendFor<T> for Cpu {
    type Backend = crate::cpu::CpuBackendImpl<T, Cpu>;
}

#[cfg(feature = "wgpu")]
use incin_core::prelude::{Wgpu, WgpuN};

#[cfg(feature = "wgpu")]
impl<T: DType> sealed::Sealed<T> for Wgpu {}

#[cfg(feature = "wgpu")]
impl<T: DType> BackendFor<T> for Wgpu {
    type Backend = crate::wgpu::WgpuBackendImpl<T, Wgpu>;
}

#[cfg(feature = "wgpu")]
impl<T: DType, N> sealed::Sealed<T> for WgpuN<N>
where
    N: incin_core::typenum::Unsigned
        + 'static
        + Send
        + Sync
        + Clone
        + Eq
        + PartialEq
        + core::fmt::Debug,
{}

#[cfg(feature = "wgpu")]
impl<T: DType, N> BackendFor<T> for WgpuN<N>
where
    N: incin_core::typenum::Unsigned
        + 'static
        + Send
        + Sync
        + Clone
        + Eq
        + PartialEq
        + core::fmt::Debug,
{
    type Backend = crate::wgpu::WgpuBackendImpl<T, WgpuN<N>>;
}

#[cfg(feature = "cuda")]
use incin_core::prelude::{Cuda, CudaN};

#[cfg(feature = "cuda")]
impl<T: DType> sealed::Sealed<T> for Cuda {}

#[cfg(feature = "cuda")]
impl<T: DType> BackendFor<T> for Cuda {
    type Backend = crate::cuda::CudaBackendImpl<T, Cuda>;
}

#[cfg(feature = "cuda")]
impl<T: DType, N> sealed::Sealed<T> for CudaN<N>
where
    N: incin_core::typenum::Unsigned
        + 'static
        + Send
        + Sync
        + Clone
        + Eq
        + PartialEq
        + core::fmt::Debug,
{}

#[cfg(feature = "cuda")]
impl<T: DType, N> BackendFor<T> for CudaN<N>
where
    N: incin_core::typenum::Unsigned
        + 'static
        + Send
        + Sync
        + Clone
        + Eq
        + PartialEq
        + core::fmt::Debug,
{
    type Backend = crate::cuda::CudaBackendImpl<T, CudaN<N>>;
}

impl<T: DType> sealed::Sealed<T> for Dyn {}

impl<T: DType> BackendFor<T> for Dyn {
    type Backend = crate::dispatch::DispatchBackend<T, Dyn>;
}

macro_rules! impl_transfer {
    ($source:ty) => {
        impl<T: DType, D: Device, NewD> incin_core::prelude::TransferTo<NewD> for $source
        where
            NewD: BackendFor<T>,
        {
            type Output = crate::IncinBackend<T, NewD>;

            fn transfer_storage<K: DType>(
                storage: &Self::Storage<K>,
                dtype: &K::Field,
                device: &NewD::Field,
            ) -> incin_core::prelude::Result<<Self::Output as Backend>::Storage<K>>
            where
                Self::Output: incin_core::prelude::SupportsDType<K>,
            {
                use incin_core::prelude::{Error, SupportsDType};
                let expected_dtype = K::to_incin(dtype);
                let source_dtype = Self::storage_dtype::<K>(storage).ok_or(
                    Error::UnsupportedBackendOperation {
                        op: "transfer_storage_metadata",
                        backend: core::any::type_name::<Self>(),
                    },
                )?;
                if source_dtype != expected_dtype {
                    return Err(Error::DTypeStorageMismatch {
                        expected: expected_dtype,
                        got: source_dtype,
                    });
                }
                Self::storage_device::<K>(storage).ok_or(Error::UnsupportedBackendOperation {
                    op: "transfer_storage_metadata",
                    backend: core::any::type_name::<Self>(),
                })?;
                let destination = NewD::to_incin(device)?;
                let dtype_id =
                    <Self::Output as SupportsDType<K>>::resolve_dtype(dtype, &destination)?;
                let shape = Self::shape::<K>(storage);
                let bytes = Self::to_bytes::<K>(storage)?;
                <Self::Output as Backend>::from_bytes::<K>(&bytes, &shape, dtype_id, &destination)
            }

            fn transfer_var(
                variable: &Self::RawVar,
                dtype: &<T as DType>::Field,
                device: &NewD::Field,
            ) -> incin_core::prelude::Result<<Self::Output as Backend>::RawVar>
            where
                Self::Output: incin_core::prelude::SupportsDType<T>,
            {
                use incin_core::prelude::SupportsDType;
                let source = Self::var_as_tensor::<T>(variable)?;
                let expected_dtype = T::to_incin(dtype);
                if let Some(got) = Self::storage_dtype::<T>(&source)
                    && got != expected_dtype
                {
                    return Err(incin_core::prelude::Error::DTypeStorageMismatch {
                        expected: expected_dtype,
                        got,
                    });
                }
                let destination = NewD::to_incin(device)?;
                let dtype_id =
                    <Self::Output as SupportsDType<T>>::resolve_dtype(dtype, &destination)?;
                let shape = Self::shape::<T>(&source);
                let bytes = Self::to_bytes::<T>(&source)?;
                let storage = <Self::Output as Backend>::from_bytes::<T>(
                    &bytes,
                    &shape,
                    dtype_id,
                    &destination,
                )?;
                <Self::Output as Backend>::var_from_tensor(&storage)
            }
        }
    };
}

#[cfg(feature = "cpu")]
impl_transfer!(crate::cpu::CpuBackendImpl<T, D>);
#[cfg(feature = "wgpu")]
impl_transfer!(crate::wgpu::WgpuBackendImpl<T, D>);
#[cfg(feature = "cuda")]
impl_transfer!(crate::cuda::CudaBackendImpl<T, D>);
impl_transfer!(crate::dispatch::DispatchBackend<T, D>);

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "cpu")]
    use incin_core::prelude::{
        DTypeId, DeviceId, Error, Grad, LayerNorm, Linear, RequiresGrad, Tensor, ToDevice, typenum,
    };

    fn assert_backend<B: Backend>() {}

    #[cfg(feature = "cpu")]
    #[test]
    fn selects_cpu_backend() {
        assert_backend::<crate::IncinBackend<f32, Cpu>>();
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn selects_wgpu_backend() {
        assert_backend::<crate::IncinBackend<f32, incin_core::prelude::Wgpu>>();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn selects_cuda_backend() {
        assert_backend::<crate::IncinBackend<f32, incin_core::prelude::Cuda>>();
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn runtime_dispatch_selects_cpu_and_preserves_metadata() {
        type B = crate::IncinBackend<Dyn, Dyn>;
        let tensor = Tensor::<Dyn, B, Dyn>::zeros(([2, 3], DTypeId::F64, DeviceId::cpu())).unwrap();
        assert_eq!(tensor.dims(), vec![2, 3]);
        assert_eq!(tensor.dtype(), DTypeId::F64);
        assert_eq!(tensor.device().unwrap(), DeviceId::cpu());
        assert!(matches!(
            tensor.inner(),
            crate::dispatch::DispatchStorage::Cpu(_)
        ));
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn checked_storage_wrapping_rejects_metadata_mismatches() {
        type B = crate::IncinBackend<Dyn, Dyn>;
        let storage = Tensor::<Dyn, B, Dyn>::zeros(([1], DTypeId::F32, DeviceId::cpu()))
            .unwrap()
            .into_inner();

        let dtype_error = Tensor::<Dyn, B, Dyn>::try_from_storage(
            storage.clone(),
            vec![1],
            DTypeId::F64,
            DeviceId::cpu(),
            <Grad as RequiresGrad>::init(()),
        )
        .unwrap_err();
        assert!(matches!(
            dtype_error,
            Error::DTypeStorageMismatch {
                expected: DTypeId::F64,
                got: DTypeId::F32,
            }
        ));

        let device_error = Tensor::<Dyn, B, Dyn>::try_from_storage(
            storage,
            vec![1],
            DTypeId::F32,
            DeviceId::wgpu(0),
            <Grad as RequiresGrad>::init(()),
        )
        .unwrap_err();
        assert!(matches!(device_error, Error::DeviceStorageMismatch { .. }));
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn runtime_from_bytes_rejects_wrong_length() {
        type B = crate::IncinBackend<Dyn, Dyn>;
        let error =
            Tensor::<Dyn, B, Dyn>::from_bytes(&[0; 3], ([1], DTypeId::F32, DeviceId::cpu()))
                .unwrap_err();
        assert!(matches!(
            error,
            Error::InvalidByteLength {
                expected: 4,
                got: 3,
            }
        ));
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn runtime_dispatch_reduces_and_transfers_through_host_bytes() {
        type B = crate::IncinBackend<Dyn, Dyn>;
        let tensor = Tensor::<Dyn, B, Dyn>::ones(([2, 2], DTypeId::F32, DeviceId::cpu())).unwrap();
        let reduced = tensor.clone().sum_all().unwrap();
        assert_eq!(reduced.to_scalar::<f32>().unwrap(), 4.0);

        let transferred = Tensor::to_device::<Dyn>(&tensor, &DeviceId::cpu()).unwrap();
        assert_eq!(transferred.dims(), vec![2, 2]);
        assert_eq!(transferred.dtype(), DTypeId::F32);
        assert_eq!(transferred.device().unwrap(), DeviceId::cpu());
        assert_eq!(
            transferred.sum_all().unwrap().to_scalar::<f32>().unwrap(),
            4.0
        );
    }

    #[cfg(all(feature = "cpu", not(feature = "wgpu")))]
    #[test]
    fn runtime_dispatch_reports_disabled_backend() {
        type B = crate::IncinBackend<Dyn, Dyn>;
        let error =
            Tensor::<Dyn, B, Dyn>::zeros(([1], DTypeId::F32, DeviceId::wgpu(0))).unwrap_err();
        assert!(matches!(
            error,
            Error::BackendUnavailable { backend: "Wgpu" }
        ));
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn static_cpu_transfer_rebuilds_dynamic_dispatch_storage() {
        type Source = crate::IncinBackend<f32, Cpu>;
        type Target = crate::IncinBackend<f32, Dyn>;
        let tensor = Tensor::<Dyn, Source>::from_slice(&[1.0f32, 2.0, 3.0], [3]).unwrap();
        let transferred = Tensor::to_device::<Dyn>(&tensor, &DeviceId::cpu()).unwrap();
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
        type B = crate::IncinBackend<Dyn, Dyn>;
        let linear = Linear::<Dyn, B>::build((10, 20, DTypeId::F32, DeviceId::cpu())).unwrap();
        assert_eq!(linear.weight.shape_dims(), vec![20, 10]);

        let biased =
            Linear::<Dyn, B, Dyn>::build((10, 20, DTypeId::F32, DeviceId::cpu(), true)).unwrap();
        assert!(biased.bias.is_some());

        let norm = LayerNorm::<Dyn, B>::build((20, DTypeId::F32, DeviceId::cpu(), 1e-5)).unwrap();
        assert_eq!(norm.weight.shape_dims(), vec![20]);
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn derived_module_transfer_changes_its_backend_output_type() {
        type Source = crate::IncinBackend<f32, Cpu>;
        type Target = crate::IncinBackend<f32, Dyn>;
        let layer = Linear::<(typenum::U2, typenum::U3), Source>::build(()).unwrap();
        let transferred = ToDevice::<Source, Dyn>::to_device(layer, &DeviceId::cpu()).unwrap();
        fn assert_target(_: &Linear<(typenum::U2, typenum::U3), Target>) {}
        assert_target(&transferred);
        assert!(matches!(
            transferred.weight.as_tensor().unwrap().inner(),
            crate::dispatch::DispatchStorage::Cpu(_)
        ));
    }
}
