//! Canonical-Only Backend Acceptance Contract Test.
//! Proves that ordinary Tensor methods (add, eq, etc.) do NOT require
//! legacy backend traits (NumericOps, FloatOps, TensorOps), but execute through
//! exact operation-keyed `Execute<Op>` bounds.

#![cfg(feature = "target-api")]

use incin_core::backend_authoring::operations::op;
use incin_core::backend_authoring::{
    AutogradBackend, Backend, Execute, ExecutionRequest, StorageBackend, SupportsDType,
    VariableBackend,
};
use incin_core::exec::{
    Alignment, Capabilities, CapabilityQuery, SupportLevel, TensorMeta, UnsupportedReason,
};
use incin_core::prelude::*;
use incin_core::shapes::ShapeBuf;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CanonicalOnlyBackend;

#[derive(Clone, Debug)]
pub struct CanonicalStorage<K: DType> {
    meta: TensorMeta,
    _phantom: core::marker::PhantomData<K>,
}

impl<K: DType> incin_core::backend_authoring::StorageOutput for CanonicalStorage<K> {}

impl StorageBackend for CanonicalOnlyBackend {
    const BACKEND_NAME: &'static str = "CanonicalOnlyBackend";
    type Storage<K: DType> = CanonicalStorage<K>;
    type Device = Cpu;

    fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta {
        &storage.meta
    }
}

impl Backend for CanonicalOnlyBackend {
    type InnerBackend = Self;

    fn to_bytes<K: DType>(_storage: &Self::Storage<K>) -> Result<Vec<u8>> {
        Ok(vec![])
    }

    fn from_bytes<K: DType>(
        _bytes: &[u8],
        dims: &[usize],
        dtype: DTypeDescriptor,
        _device: &DeviceId,
    ) -> Result<Self::Storage<K>> {
        let meta = TensorMeta::contiguous(
            ShapeBuf::from_slice(dims),
            dtype,
            DeviceId::CPU,
            Alignment::new(1)?,
            dims.iter().product(),
        )?;
        Ok(CanonicalStorage {
            meta,
            _phantom: core::marker::PhantomData,
        })
    }

}

impl VariableBackend for CanonicalOnlyBackend {
    type RawVar = ();

    fn var_as_tensor<K: DType>(_var: &Self::RawVar) -> Result<Self::Storage<K>> {
        Err(Error::Backend(BackendError::Unsupported {
            backend: Self::BACKEND_NAME,
            reason: UnsupportedReason::MissingDeviceFeature {
                feature: "var_as_tensor",
            },
        }))
    }

    fn var_from_tensor<K: DType>(_tensor: &Self::Storage<K>) -> Result<Self::RawVar> {
        Ok(())
    }

    fn assign_var<K: DType>(_var: &mut Self::RawVar, _tensor: &Self::Storage<K>) -> Result<()> {
        Ok(())
    }
}

impl AutogradBackend for CanonicalOnlyBackend {
    type Grads = ();

    fn backward<K: DType>(_tensor: &Self::Storage<K>) -> Result<Self::Grads> { Ok(()) }

    fn get_grad<K: DType>(
        _tensor: &Self::Storage<K>,
        _grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>> { Ok(None) }
}

impl Capabilities for CanonicalOnlyBackend {
    fn support(&self, _query: &CapabilityQuery) -> SupportLevel {
        SupportLevel::Native
    }
}

impl<K: DType> SupportsDType<K> for CanonicalOnlyBackend {
    fn resolve_dtype(field: &K::Field, _device: &DeviceId) -> Result<DTypeDescriptor> {
        Ok(K::descriptor(field))
    }
}

impl Execute<op::Add> for CanonicalOnlyBackend {
    type Output = CanonicalStorage<f32>;

    fn execute(
        &self,
        _request: ExecutionRequest<'_, op::Add, Self>,
    ) -> core::result::Result<Self::Output, BackendError> {
        let meta = TensorMeta::contiguous(
            ShapeBuf::from_slice(&[3]),
            <f32 as ConstDType>::DESCRIPTOR,
            DeviceId::CPU,
            Alignment::new(1).unwrap(),
            3,
        )
        .unwrap();
        Ok(CanonicalStorage {
            meta,
            _phantom: core::marker::PhantomData,
        })
    }
}

impl Execute<op::CmpEq> for CanonicalOnlyBackend {
    type Output = CanonicalStorage<bool>;

    fn execute(
        &self,
        _request: ExecutionRequest<'_, op::CmpEq, Self>,
    ) -> core::result::Result<Self::Output, BackendError> {
        let meta = TensorMeta::contiguous(
            ShapeBuf::from_slice(&[3]),
            <bool as ConstDType>::DESCRIPTOR,
            DeviceId::CPU,
            Alignment::new(1).unwrap(),
            3,
        )
        .unwrap();
        Ok(CanonicalStorage {
            meta,
            _phantom: core::marker::PhantomData,
        })
    }
}

#[test]
fn test_canonical_only_backend_add_and_eq_compile() -> Result<()> {
    let meta_f32 = TensorMeta::contiguous(
        ShapeBuf::from_slice(&[3]),
        <f32 as ConstDType>::DESCRIPTOR,
        DeviceId::CPU,
        Alignment::new(1)?,
        3,
    )?;

    let s1 = CanonicalStorage::<f32> {
        meta: meta_f32.clone(),
        _phantom: core::marker::PhantomData,
    };
    let s2 = CanonicalStorage::<f32> {
        meta: meta_f32,
        _phantom: core::marker::PhantomData,
    };

    let a = Tensor::<Dyn, CanonicalOnlyBackend, f32, NoGrad>::try_from_storage(
        s1,
        ShapeBuf::from_slice(&[3]),
        core::marker::PhantomData,
        core::marker::PhantomData,
        core::marker::PhantomData,
    )?;
    let b = Tensor::<Dyn, CanonicalOnlyBackend, f32, NoGrad>::try_from_storage(
        s2,
        ShapeBuf::from_slice(&[3]),
        core::marker::PhantomData,
        core::marker::PhantomData,
        core::marker::PhantomData,
    )?;

    let _c = a.add(&b)?;
    let _mask: Tensor<Dyn, CanonicalOnlyBackend, bool, NoGrad> = a.eq(&b)?;

    Ok(())
}
