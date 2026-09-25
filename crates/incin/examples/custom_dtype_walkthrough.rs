//! Custom dtypes end to end: what D-110 unsealed, and what stays shut.
//!
//! This is recommendation #1 of
//! `docs/plan/research/0.2.0/novel-solutions-platform.md` (the top
//! unlock-per-effort item) in action: the custom-dtype authoring checklist
//! from friction-walk B, demonstrated against decision D-110 (unsealed
//! `TensorElement`) from `docs/plan/research/0.2.0/96-extension-points.md`.
//!
//! The walkthrough defines a custom 3-byte posit-like dtype entirely
//! in-example — the way it would live in a downstream crate, with no patch
//! to `incin-core`, no `BuiltinDType` impl, and no `DTypeId` — and takes it
//! through each gate deliberately:
//!
//! 1. The unsealed `TensorElement` bound admits the newtype, so `PlainDType`
//!    is implementable out of tree (this file fails to compile with the seal
//!    in place). Element definition mirrors
//!    `crates/incin-core/tests/custom_element_proof.rs`.
//! 2. Element access works against a backend that advertises the descriptor:
//!    `from_slice` constructs, `dtype()` reports the custom descriptor,
//!    `builtin_dtype_id()` is `None`, and `to_vec_elem()` round-trips.
//! 3. Each still-closed gate refuses loudly: the distributed static plan at
//!    compile time (`BuiltinDType` bound — shown as a non-compiling snippet
//!    in comments so this file still builds), the safetensors export and the
//!    CPU backend's kernel admission at runtime with typed errors.
//! 4. The `DeviceKey` side: an `External(DeviceKey)` identity plus a digest
//!    distinction demo.
//!
//! Run with:
//! `cargo run -p incin --example custom_dtype_walkthrough --no-default-features --features incin-backends/cpu,incin/cpu`

#![cfg(feature = "cpu")]

use core::marker::PhantomData;
use std::sync::Arc;

use incin::prelude::*;
use incin_backends::cpu::CpuBackendImpl;
use incin_core::backend_authoring::{
    Backend, Capabilities, CapabilityQuery, Execute, ExecutionRequest, HostInterop, HostReadback,
    OperationIdentity, ShapeBuf, StorageBackend, StorageOutput, SupportLevel, SupportsDType,
    TensorMeta, UnsupportedReason, VariableBackend, op,
};
use incin_core::nn::{StatePath, StateVisitor, VisitState, save_safetensors};
use incin_core::shapes::OperationKind;
use incin_core::tensor::dtype::StorageEncoding;

// ============================================================================
// Gate 1: the downstream dtype — a 3-byte posit-like scalar + logical dtype
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
/// `BuiltinDType` impl. That absence is the point being walked through.
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

/// Shorthand for the walkthrough's tensor: four posit24 values on Acme.
type PositTensor = Tensor<s![4], AcmeBackend, Posit24>;

/// Shorthand for the walkthrough's model weight over the same tensor.
type PositWeights = Buffer<s![4], AcmeBackend, Posit24>;

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

// A variable handle so the custom tensor can sit in a model buffer and reach
// the state exporter. The handle is just shared storage; the refusal we are
// walking toward happens later, at the closed safetensors vocabulary.
#[derive(Debug, Clone)]
struct AcmeVar(Arc<std::sync::RwLock<AcmeStorage>>);

impl VariableBackend for AcmeBackend {
    type Var<K: DType> = AcmeVar;

    fn var_from_tensor<K: DType>(storage: &Self::Storage<K>) -> Result<Self::Var<K>> {
        Ok(AcmeVar(Arc::new(std::sync::RwLock::new(storage.clone()))))
    }

    fn var_as_tensor<K: DType>(var: &Self::Var<K>) -> Result<Self::Storage<K>> {
        Ok(var.0.read().unwrap().clone())
    }

    fn assign_var<K: DType>(var: &mut Self::Var<K>, storage: &Self::Storage<K>) -> Result<()> {
        *var.0.write().unwrap() = storage.clone();
        Ok(())
    }
}

// A one-weight model holding the custom tensor, so the safetensors exporter
// has something to refuse.
struct PositModel {
    weights: PositWeights,
}

impl VisitState<AcmeBackend> for PositModel {
    fn visit_state<V: StateVisitor<AcmeBackend>>(
        &self,
        path: &StatePath,
        visitor: &mut V,
    ) -> Result<()> {
        visitor.visit_buffer(path, &self.weights)
    }
}

// ============================================================================
// The walkthrough
// ============================================================================

fn main() -> incin::Result<()> {
    section("1. The downstream dtype: D-110 admits the 3-byte scalar");
    gate_dtype_identity();

    section("2. Element access works on a backend that advertises it");
    let tensor = gate_element_access()?;

    section("3. The gates that stay closed refuse loudly");
    gate_closed_compile_time_note();
    gate_closed_cpu_kernel()?;
    gate_closed_safetensors_export(&tensor)?;

    section("4. DeviceKey side: External(DeviceKey) identity");
    gate_device_identity();

    println!("\nWalkthrough complete: custom dtype constructs and reads back,");
    println!("and every closed gate refused with a typed error. See the");
    println!("checklist summary at the end of this file for the full map.");

    Ok(())
}

/// Gate 1: the unsealed bound admits the POD newtype, so `PlainDType` is
/// implementable out of tree. Both asserts fail to compile with the seal in
/// place; neither needs a `BuiltinDType` impl.
fn gate_dtype_identity() {
    assert_element::<PositElem>();
    assert_plain::<Posit24>();
    assert_eq!(Posit24::DESCRIPTOR.builtin_id(), None);
    assert_eq!(Posit24::DESCRIPTOR.key().namespace(), "acme");
    println!(
        "  PositElem satisfies TensorElement (NoUninit + Zeroable + Copy + Debug + Send + Sync)"
    );
    println!("  Posit24 satisfies PlainDType with Elem = PositElem, no DTypeId, no BuiltinDType");
    println!("  descriptor: {:?}", Posit24::DESCRIPTOR);
}

/// Gate 2: the dtype behaves like a first-class scalar — construct from a
/// slice, read the values back, print them.
fn gate_element_access() -> incin::Result<PositTensor> {
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
    assert_eq!(tensor.device()?, DeviceId::cpu());
    println!(
        "  constructed: shape {:?}, dtype {}, device {:?}",
        tensor.dims(),
        tensor.dtype().name(),
        tensor.device()?
    );

    let back = tensor.to_vec_elem().expect("element extraction reads back");
    assert_eq!(back, elems);
    println!("  read back: {back:?}");
    println!("  builtin_dtype_id() is None: the tensor is custom all the way down");
    Ok(tensor)
}

/// Gate 3a (compile time): the distributed static plan stays closed. This
/// snippet does NOT compile, so it lives in a comment — the refusal is the
/// `BuiltinDType` bound, and the error below is what rustc reports:
/*
    use incin_backends::dist::{CollectiveTuningProblem, TuneAllGather};

    let _ = CollectiveTuningProblem::new_static::<Posit24, U16, TuneAllGather>(
        GroupId::new(1, 2).unwrap(),
        topology,
        Determinism::Permitted,
        1024,
    );
    // error[E0277]: the trait bound `Posit24: BuiltinDType` is not satisfied
    // (`new_static` requires `K: ConstDType + BuiltinDType + CollectiveDType`
    // because the tuning cache keys measurements by built-in `DTypeId`).
    // In-tree proof: crates/incin-backends/tests/collective_tuning_compile_fail/
    // custom_dtype_missing_builtin_id.rs.
*/
fn gate_closed_compile_time_note() {
    println!("  distributed static plan: still requires BuiltinDType at compile time");
    println!("  (CollectiveTuningProblem::new_static::<Posit24, ..> does not compile;");
    println!("  the non-compiling snippet is quoted in this function's docs source)");
}

/// Gate 3b (runtime): `from_slice` now compiles for the custom dtype (the
/// `PlainDType` bound is satisfied), and the CPU backend's admission refuses
/// it carrying the exact descriptor — the previously-impossible path fails
/// loudly instead of miscompiling or coercing.
fn gate_closed_cpu_kernel() -> incin::Result<()> {
    let elems = [PositElem([9, 9, 9]); 4];
    match Tensor::<s![4], CpuBackendImpl, Posit24>::from_slice(&elems, ()) {
        Ok(_) => println!("  UNEXPECTED: the CPU backend accepted acme/posit24"),
        Err(error) => {
            match &error {
                Error::UnsupportedDType { dtype, .. } => {
                    assert_eq!(*dtype, Posit24::DESCRIPTOR);
                }
                other => panic!("expected a typed dtype refusal, got {other:?}"),
            }
            println!("  CPU kernel admission refused the custom dtype: {error}");
        }
    }
    Ok(())
}

/// Gate 3c (runtime): the safetensors snapshot export matches a closed
/// builtin table (`safetensors_dtype` in `crates/incin-core/src/serialize.rs`)
/// and returns a typed error for anything else. The custom tensor reaches the
/// exporter as a model weight, and the export names the dtype on refusal.
fn gate_closed_safetensors_export(tensor: &PositTensor) -> incin::Result<()> {
    let var = AcmeBackend::var_from_tensor::<Posit24>(tensor.inner())?;
    let weights = PositWeights::from_parts_checked(
        var,
        ShapeBuf::from_slice(&[4]),
        Posit24::init(()),
        <Cpu as Device>::init(()),
    )?;
    let model = PositModel { weights };
    let path = std::env::temp_dir().join("incin-custom-dtype-walkthrough.safetensors");
    match save_safetensors::<AcmeBackend, PositModel, _>(&model, &path) {
        Ok(()) => println!("  UNEXPECTED: safetensors accepted acme/posit24"),
        Err(error) => println!("  safetensors export refused the custom dtype: {error}"),
    }
    std::fs::remove_file(&path).ok();
    Ok(())
}

/// Gate 4: third-party device identity is open too — a structured
/// `DeviceKey` (namespace + name + version) carried by
/// `DeviceKind::External`, so adding a backend needs no framework patch.
fn gate_device_identity() {
    let key = DeviceKey::new("acme", "npu", 1);
    let id = DeviceId::external(key, 0);
    assert_eq!(id.kind(), DeviceKind::External(key));
    assert_eq!(id.kind().name(), "npu");
    assert_eq!(id.ordinal(), 0);
    println!(
        "  external device: {id:?} (kind name {:?})",
        id.kind().name()
    );

    // Digest distinction: two vendors shipping an "npu" are different
    // families even though `name()` reports the same short device name for
    // both — a device set mixing them is refused at runtime.
    let other = DeviceId::external(DeviceKey::new("emca", "npu", 1), 0);
    assert_ne!(id.kind(), other.kind());
    match DeviceSet::new([id, other]) {
        Ok(_) => println!("  UNEXPECTED: one set held two vendors' npus"),
        Err(error) => println!("  device set refused the mixed vendors: {error}"),
    }
    let pair = DeviceSet::new([id, DeviceId::external(key, 1)])
        .expect("two ordinals of one external family are a valid set");
    println!(
        "  same key at two ordinals is one family: len {}",
        pair.len()
    );
}

fn section(title: &str) {
    println!("\n{title}");
    println!("{}", "-".repeat(title.len()));
}

// ============================================================================
// Checklist summary (recommendation #1: the custom-dtype authoring checklist)
// ============================================================================
//
// What worked (D-110 unsealed, all in this file, no framework patch):
// - `PositElem` (3-byte POD newtype) satisfies `TensorElement` via the
//   blanket impl over `NoUninit + Zeroable + Copy + Debug + Send + Sync`.
// - `Posit24` implements `DType + ConstDType + PlainDType` with a stable
//   `DTypeKey("acme", "posit24", 1)`; `builtin_id()` is `None`.
// - Element access on the descriptor-advertising backend: `from_slice`
//   constructs, `dtype()` reports the custom descriptor, `to_vec_elem()`
//   round-trips, and the tensor wraps as a `Buffer` for state traversal.
//
// What refuses, and the exact error proving it:
// - Distributed static plan: COMPILE-TIME refusal. `Posit24` does not
//   implement `BuiltinDType`, so `CollectiveTuningProblem::new_static`
//   (`K: ConstDType + BuiltinDType + CollectiveDType`) and the static
//   planners (`K: ConstDType + BuiltinDType + HybridPlanDType`) reject it
//   before anything runs. Quoted, not compiled, in
//   `gate_closed_compile_time_note`.
// - CPU kernel execution: RUNTIME refusal, printed above —
//   `Error::UnsupportedDType` carrying the `acme/posit24` descriptor.
// - Safetensors export: RUNTIME refusal, printed above — the closed builtin
//   table errors naming the custom dtype (the postcard state envelope beside
//   it is descriptor-driven and would round-trip it).
// - Mixed-vendor device set: RUNTIME refusal, printed above —
//   `DeviceSetError::Mixed` even though both vendors name their device "npu".
//
// Where PROPOSALS D-110 governs:
// - The `TensorElement` seal is removed; the POD bound is the proof
//   obligation. `BuiltinDType` gates stay in the four closed subsystems
//   (distributed static planners + plan digests, collective tuning keys,
//   kernel-table fast paths, safetensors export) during the transition.
// - `External(DeviceKey)` (namespace + name + version, paralleling
//   `DTypeKey`) is the third-party device tier; maintained backends keep
//   first-class `DeviceKind` variants. See `PROPOSALS.md` D-110 and
//   `docs/plan/research/0.2.0/96-extension-points.md` (decision section).
