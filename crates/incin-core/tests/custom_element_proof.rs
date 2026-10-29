//! Executable proof for issue #96 / D-110: a custom dtype defined entirely
//! outside `incin-core` gets element access, while the closed subsystems
//! still refuse it with typed errors.
//!
//! Everything custom here (`PositElem`, `Posit24`, `AcmeBackend`) lives in
//! this file, the way it would live in a downstream crate: no patch to
//! `incin-core`, no `BuiltinDType` impl, no `DTypeId`. What the test proves:
//!
//! 1. The unsealed [`TensorElement`](incin_core::prelude::TensorElement) bound
//!    admits a 3-byte posit-like POD newtype, so `PlainDType` is implementable
//!    out of tree (this file fails to compile with the seal in place).
//! 2. Element access works against a backend that advertises the descriptor:
//!    `from_slice` constructs, `dtype()` reports the custom descriptor,
//!    `builtin_dtype_id()` is `None`, and `to_vec_elem()` round-trips.
//! 3. The gates that stay closed refuse the same dtype at runtime with typed
//!    errors: the CPU backend's dtype admission (`SupportsDType`) and the
//!    checkpoint manifest's closed dtype vocabulary (`CheckpointDType`).

#![cfg(feature = "std")]

extern crate incin_core as incin;

use core::marker::PhantomData;

use incin_backends::cpu::CpuBackendImpl;
use incin_core::backend_authoring::{
    Backend, Capabilities, CapabilityQuery, Execute, ExecutionRequest, HostInterop, HostReadback,
    OperationIdentity, StorageBackend, StorageOutput, SupportLevel, SupportsDType, TensorMeta,
    UnsupportedReason, op,
};
use incin_core::error::{BackendError, Error};
use incin_core::nn::CheckpointDType;
use incin_core::prelude::*;
use incin_core::tensor::dtype::StorageEncoding;
use incin_macros::s;

// ============================================================================
// The downstream dtype: a 3-byte posit-like scalar + its logical dtype
// ============================================================================

/// A 24-bit posit-like element. Every byte pattern is a valid value, so host
/// extraction round-trips whatever the backend stored.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PositElem([u8; 3]);

// SAFETY: `repr(transparent)` over `[u8; 3]`; all-zero and every other byte
// pattern is a valid value.
unsafe impl bytemuck::NoUninit for PositElem {}
// SAFETY: same layout argument; zeroed memory is a valid value.
unsafe impl bytemuck::Zeroable for PositElem {}

/// The downstream logical dtype. Note what is absent: no `DTypeId`, no
/// `BuiltinDType` impl. That absence is the point being proved.
#[derive(Clone, Debug, PartialEq)]
struct Posit24;

impl DType for Posit24 {
    type Arg = ();
    type Field = PhantomData<Self>;

    fn init(_: ()) -> Self::Field {
        PhantomData
    }

    fn descriptor(_: &Self::Field) -> DTypeDescriptor {
        Self::DESCRIPTOR
    }
}

impl ConstDType for Posit24 {
    const DESCRIPTOR: DTypeDescriptor = DTypeDescriptor::new(
        DTypeKey::new("acme", "posit24", 1),
        DTypeKind::Opaque,
        StorageEncoding::scalar(3, 1),
    );
}

impl PlainDType for Posit24 {
    type Elem = PositElem;
}

fn assert_element<T: TensorElement>() {}
fn assert_plain<K: PlainDType>() {}

// ============================================================================
// The downstream backend: a byte store that admits any descriptor
// ============================================================================

const ACME_NAME: &str = "Acme";

#[derive(Debug, Clone, Default)]
struct AcmeBackend;

#[derive(Debug, Clone)]
struct AcmeStorage {
    meta: TensorMeta,
    bytes: Vec<u8>,
}

impl StorageBackend for AcmeBackend {
    const BACKEND_NAME: &'static str = ACME_NAME;
    type Storage<K: DType> = AcmeStorage;
    type Device = Cpu;

    fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta {
        &storage.meta
    }
}

impl StorageOutput for AcmeStorage {}

impl Capabilities for AcmeBackend {
    fn support(&self, query: &CapabilityQuery) -> SupportLevel {
        match &query.operation {
            OperationIdentity::Builtin(OperationKind::TensorFromData) => SupportLevel::Native,
            OperationIdentity::Builtin(operation) => {
                SupportLevel::Unsupported(UnsupportedReason::Operation {
                    operation: *operation,
                })
            }
            OperationIdentity::Custom(operation) => {
                SupportLevel::Unsupported(UnsupportedReason::CustomOperation {
                    operation: operation.clone(),
                })
            }
        }
    }
}

impl<K: DType> SupportsDType<K> for AcmeBackend {
    fn resolve_dtype(field: &K::Field, _device: &DeviceId) -> Result<DTypeDescriptor> {
        Ok(K::descriptor(field))
    }
}

impl Backend for AcmeBackend {
    type InnerBackend = Self;
}

impl HostReadback for AcmeBackend {
    fn float_to_vec1<K: DType>(storage: &AcmeStorage) -> Result<Vec<f64>> {
        Err(Error::UnsupportedDType {
            dtype: storage.meta.dtype(),
            backend: ACME_NAME,
            op: "readback",
        })
    }

    fn int_to_vec1<K: DType>(storage: &AcmeStorage) -> Result<Vec<i64>> {
        Err(Error::UnsupportedDType {
            dtype: storage.meta.dtype(),
            backend: ACME_NAME,
            op: "readback",
        })
    }
}

impl HostInterop for AcmeBackend {
    fn to_bytes<K: DType>(storage: &AcmeStorage) -> Result<Vec<u8>> {
        Ok(storage.bytes.clone())
    }

    fn from_bytes<K: DType>(
        bytes: &[u8],
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<AcmeStorage> {
        let elements = shape
            .iter()
            .try_fold(1usize, |count, dim| count.checked_mul(*dim))
            .ok_or(Error::InvalidByteLength {
                expected: usize::MAX,
                got: bytes.len(),
            })?;
        let expected =
            dtype.size_bytes(elements, incin_core::shapes::error::OperationKind::Storage)?;
        if bytes.len() != expected {
            return Err(Error::InvalidByteLength {
                expected,
                got: bytes.len(),
            });
        }
        let meta = TensorMeta::contiguous(
            ShapeBuf::from_slice(shape),
            dtype,
            *device,
            incin_core::exec::Alignment::BYTE,
            elements,
        )
        .map_err(|error| Error::Msg(format!("acme metadata refused the allocation: {error}")))?;
        Ok(AcmeStorage {
            meta,
            bytes: bytes.to_vec(),
        })
    }
}

impl Execute<op::TensorFromData> for AcmeBackend {
    type Output = AcmeStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, op::TensorFromData, Self>,
    ) -> core::result::Result<AcmeStorage, BackendError> {
        const OPERATION: OperationKind = OperationKind::TensorFromData;
        if !request.inputs.is_empty() {
            return Err(BackendError::InvalidInput {
                operation: OPERATION,
                reason: "data creation takes no operand",
            });
        }
        let attributes = request.operation.descriptor().attributes();
        let bytes = request.payload.ok_or(BackendError::InvalidInput {
            operation: OPERATION,
            reason: "data creation requires borrowed bytes",
        })?;
        Self::from_bytes::<f32>(
            bytes,
            &attributes.shape,
            attributes.dtype,
            &attributes.device,
        )
        .map_err(|_| BackendError::InvalidInput {
            operation: OPERATION,
            reason: "acme storage refused the payload",
        })
    }
}

// ============================================================================
// Proof
// ============================================================================

#[test]
fn custom_pod_element_satisfies_the_unsealed_bound() {
    // Both lines are the seal proof: with `TensorElementSealed` in place,
    // neither bound is satisfiable outside `incin-core`.
    assert_element::<PositElem>();
    assert_plain::<Posit24>();
    assert_eq!(Posit24::DESCRIPTOR.builtin_id(), None);
    assert_eq!(Posit24::DESCRIPTOR.key().namespace(), "acme");
}

#[test]
fn custom_dtype_round_trips_through_slice_and_interop_on_its_own_backend() {
    let elems = [
        PositElem([1, 0, 0]),
        PositElem([2, 0, 0]),
        PositElem([3, 0, 0]),
        PositElem([4, 0, 0]),
    ];
    let tensor = Tensor::<s![4], AcmeBackend, Posit24>::from_slice(&elems, ())
        .expect("a backend advertising the descriptor constructs");
    assert_eq!(tensor.dtype(), Posit24::DESCRIPTOR);
    assert_eq!(tensor.builtin_dtype_id(), None);
    assert_eq!(
        tensor.to_vec_elem().expect("element extraction reads back"),
        elems
    );
}

#[test]
fn builtin_backend_refuses_the_custom_dtype_with_a_typed_error() {
    // `from_slice` now compiles for the custom dtype (the `PlainDType` bound
    // is satisfied), and the CPU backend's admission refuses it at runtime
    // carrying the exact descriptor - the previously-impossible path fails
    // loudly instead of miscompiling or coercing.
    let elems = [PositElem([9, 9, 9]); 4];
    let error = Tensor::<s![4], CpuBackendImpl, Posit24>::from_slice(&elems, ())
        .expect_err("CPU never advertised acme/posit24");
    match error {
        Error::UnsupportedDType { dtype, .. } => assert_eq!(dtype, Posit24::DESCRIPTOR),
        other => panic!("expected a typed dtype refusal, got {other:?}"),
    }
}

#[test]
fn checkpoint_manifest_refuses_the_custom_dtype_with_a_named_error() {
    // The manifest's closed dtype vocabulary is one of the gates that stays:
    // a custom descriptor is refused at runtime, naming its key.
    let error = CheckpointDType::from_descriptor(Posit24::DESCRIPTOR)
        .descriptor()
        .expect_err("the manifest names built-ins only");
    let rendered = error.to_string();
    assert!(
        rendered.contains("acme") && rendered.contains("posit24"),
        "the refusal should name the custom key, got: {rendered}"
    );
}
