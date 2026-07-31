//! Stable identities for tuning environments.
//!
//! Device ordinals are deliberately absent. An ordinal is a process-local
//! address which changes under visibility masks and can name different
//! physical devices on different hosts. Persistent tuning keys instead bind
//! the vendor identifier, architecture, driver, compiler, compiler target,
//! and semantic compiler options.

use alloc::{
    string::{String, ToString},
    vec::Vec,
};
use core::{
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
};
#[cfg(feature = "cuda")]
use incin_core::prelude::{Cuda, CudaN};
use incin_core::{
    prelude::{Cpu, DeviceKind, Dyn},
    typenum::{NonZero, Unsigned},
};

const MAX_IDENTITY_FIELD_BYTES: usize = 256;
const IDENTITY_SCHEMA: &[u8] = b"incin.tuning.identity.v1";

/// A stable backend-family tag.
///
/// This mirrors the runtime [`DeviceKind`] vocabulary while implementing
/// ordering and hashing for persistent cache keys.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum BackendIdentity {
    /// Host CPU execution.
    Cpu,
    /// NVIDIA CUDA execution.
    Cuda,
    /// WebGPU execution.
    Wgpu,
    /// Native Apple Metal execution.
    Metal,
}

impl BackendIdentity {
    /// The stable lowercase spelling included in identity digests.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Wgpu => "wgpu",
            Self::Metal => "metal",
        }
    }

    fn from_device_kind(kind: DeviceKind) -> core::result::Result<Self, IdentityError> {
        match kind {
            DeviceKind::Cpu => Ok(Self::Cpu),
            DeviceKind::Cuda => Ok(Self::Cuda),
            DeviceKind::Wgpu => Ok(Self::Wgpu),
            DeviceKind::Metal => Ok(Self::Metal),
            _ => Err(IdentityError::UnsupportedBackend),
        }
    }
}

/// A three-component software version used for drivers, compilers, and
/// transports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SoftwareVersion {
    major: u32,
    minor: u32,
    patch: u32,
}

impl SoftwareVersion {
    /// Creates a version triple.
    #[must_use]
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Returns `(major, minor, patch)`.
    #[must_use]
    pub const fn components(self) -> (u32, u32, u32) {
        (self.major, self.minor, self.patch)
    }
}

/// A failure to construct or project a stable identity.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum IdentityError {
    /// A required field was empty.
    #[error("tuning identity field `{field}` must not be empty")]
    EmptyField {
        /// The field name.
        field: &'static str,
    },
    /// A field exceeded the persistent-format bound.
    #[error("tuning identity field `{field}` is {actual} bytes; maximum is {maximum}")]
    FieldTooLong {
        /// The field name.
        field: &'static str,
        /// Supplied byte length.
        actual: usize,
        /// Maximum byte length.
        maximum: usize,
    },
    /// A field contained whitespace/control data which is not canonical.
    #[error("tuning identity field `{field}` is not canonical printable ASCII")]
    NonCanonicalField {
        /// The field name.
        field: &'static str,
    },
    /// A future runtime backend is not yet understood by this identity schema.
    #[error("runtime backend is not supported by tuning identity schema v1")]
    UnsupportedBackend,
    /// A `Dyn` identity cannot be projected to the requested static backend.
    #[error("tuning identity backend mismatch: expected {expected}, found {actual}")]
    BackendMismatch {
        /// Static backend requested by the caller.
        expected: &'static str,
        /// Runtime backend found in the identity.
        actual: &'static str,
    },
    /// A dynamic topology declared an empty world.
    #[error("tuning topology world size must be nonzero")]
    ZeroWorld,
    /// The number of rank identities did not match the declared world.
    #[error("tuning topology declares world {world} but contains {devices} rank devices")]
    WorldMismatch {
        /// Declared static or dynamic world.
        world: usize,
        /// Number of device records.
        devices: usize,
    },
    /// A dynamic topology had a different world from the requested static one.
    #[error("cannot project dynamic topology world {actual} to static world {expected}")]
    StaticWorldMismatch {
        /// Type-level world requested by the caller.
        expected: usize,
        /// Runtime world stored in the topology.
        actual: usize,
    },
    /// Two ranks named the same stable physical device.
    #[error(
        "tuning topology aliases physical device `{persistent_id}` at ranks {first_rank} and {second_rank}"
    )]
    AliasedDevice {
        /// Vendor-persistent identifier.
        persistent_id: String,
        /// First rank using the device.
        first_rank: usize,
        /// Second rank using the same device.
        second_rank: usize,
    },
    /// A link referred to a rank not in the topology.
    #[error("tuning topology link {from}->{to} is outside world {world}")]
    LinkOutOfRange {
        /// Link source rank.
        from: usize,
        /// Link destination rank.
        to: usize,
        /// Topology world.
        world: usize,
    },
    /// A link from a rank to itself is not a communication edge.
    #[error("tuning topology link {rank}->{rank} is a self link")]
    SelfLink {
        /// The repeated rank.
        rank: usize,
    },
    /// The same directed edge was recorded more than once.
    #[error("tuning topology contains duplicate link {from}->{to}")]
    DuplicateLink {
        /// Link source rank.
        from: usize,
        /// Link destination rank.
        to: usize,
    },
    /// The process layout does not cover exactly the declared ranks.
    #[error("process layout {processes}x{ranks_per_process} does not cover topology world {world}")]
    ProcessLayoutMismatch {
        /// Number of processes.
        processes: usize,
        /// Ranks assigned to each process.
        ranks_per_process: usize,
        /// Topology world.
        world: usize,
    },
    /// A CUDA runtime query failed.
    #[error("failed to query CUDA {component}: {message}")]
    CudaQuery {
        /// Driver field being queried.
        component: &'static str,
        /// Driver error.
        message: String,
    },
}

fn checked_field(field: &'static str, value: &str) -> core::result::Result<String, IdentityError> {
    if value.is_empty() {
        return Err(IdentityError::EmptyField { field });
    }
    if value.len() > MAX_IDENTITY_FIELD_BYTES {
        return Err(IdentityError::FieldTooLong {
            field,
            actual: value.len(),
            maximum: MAX_IDENTITY_FIELD_BYTES,
        });
    }
    if value.trim() != value
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() || byte == b' ')
    {
        return Err(IdentityError::NonCanonicalField { field });
    }
    Ok(value.to_string())
}

mod sealed {
    pub trait Sealed {}
}

/// A compile-time device marker with a known backend family.
///
/// This trait is sealed: static identities can only be constructed for Incin
/// device markers whose backend is known. Use `DeviceFingerprint<Dyn>` when
/// the backend itself is selected at runtime.
pub trait StaticBackend: sealed::Sealed + 'static {
    /// Backend encoded by the marker.
    const BACKEND: BackendIdentity;
}

impl sealed::Sealed for Cpu {}
impl StaticBackend for Cpu {
    const BACKEND: BackendIdentity = BackendIdentity::Cpu;
}

#[cfg(feature = "cuda")]
impl sealed::Sealed for Cuda {}
#[cfg(feature = "cuda")]
impl StaticBackend for Cuda {
    const BACKEND: BackendIdentity = BackendIdentity::Cuda;
}

#[cfg(feature = "cuda")]
impl<N: Unsigned + 'static> sealed::Sealed for CudaN<N> {}
#[cfg(feature = "cuda")]
impl<N: Unsigned + 'static> StaticBackend for CudaN<N> {
    const BACKEND: BackendIdentity = BackendIdentity::Cuda;
}

#[cfg(feature = "wgpu")]
impl sealed::Sealed for incin_core::prelude::Wgpu {}
#[cfg(feature = "wgpu")]
impl StaticBackend for incin_core::prelude::Wgpu {
    const BACKEND: BackendIdentity = BackendIdentity::Wgpu;
}

#[cfg(feature = "wgpu")]
impl<N: Unsigned + 'static> sealed::Sealed for incin_core::prelude::WgpuN<N> {}
#[cfg(feature = "wgpu")]
impl<N: Unsigned + 'static> StaticBackend for incin_core::prelude::WgpuN<N> {
    const BACKEND: BackendIdentity = BackendIdentity::Wgpu;
}

/// A stable physical-device identity.
///
/// `D` carries the backend family at compile time for static device markers.
/// `DeviceFingerprint<Dyn>` stores and checks that family at runtime. Device
/// ordinal is intentionally neither accepted nor stored.
pub struct DeviceFingerprint<D = Dyn> {
    backend: BackendIdentity,
    persistent_id: String,
    architecture: String,
    driver: SoftwareVersion,
    marker: PhantomData<fn() -> D>,
}

impl<D: StaticBackend> DeviceFingerprint<D> {
    /// Constructs an identity for a statically known backend.
    pub fn new(
        persistent_id: &str,
        architecture: &str,
        driver: SoftwareVersion,
    ) -> core::result::Result<Self, IdentityError> {
        Self::from_parts(D::BACKEND, persistent_id, architecture, driver)
    }

    /// Erases the static backend marker while retaining its runtime proof.
    #[must_use]
    pub fn erase(self) -> DeviceFingerprint<Dyn> {
        DeviceFingerprint {
            backend: self.backend,
            persistent_id: self.persistent_id,
            architecture: self.architecture,
            driver: self.driver,
            marker: PhantomData,
        }
    }
}

impl DeviceFingerprint<Dyn> {
    /// Constructs a runtime-selected identity and validates all stable fields.
    pub fn new_dyn(
        backend: DeviceKind,
        persistent_id: &str,
        architecture: &str,
        driver: SoftwareVersion,
    ) -> core::result::Result<Self, IdentityError> {
        Self::from_parts(
            BackendIdentity::from_device_kind(backend)?,
            persistent_id,
            architecture,
            driver,
        )
    }

    /// Projects a runtime identity to a statically known backend.
    pub fn try_into_static<D: StaticBackend>(
        self,
    ) -> core::result::Result<DeviceFingerprint<D>, IdentityError> {
        if self.backend != D::BACKEND {
            return Err(IdentityError::BackendMismatch {
                expected: D::BACKEND.name(),
                actual: self.backend.name(),
            });
        }
        Ok(DeviceFingerprint {
            backend: self.backend,
            persistent_id: self.persistent_id,
            architecture: self.architecture,
            driver: self.driver,
            marker: PhantomData,
        })
    }
}

impl<D> DeviceFingerprint<D> {
    fn from_parts(
        backend: BackendIdentity,
        persistent_id: &str,
        architecture: &str,
        driver: SoftwareVersion,
    ) -> core::result::Result<Self, IdentityError> {
        Ok(Self {
            backend,
            persistent_id: checked_field("persistent_id", persistent_id)?,
            architecture: checked_field("architecture", architecture)?,
            driver,
            marker: PhantomData,
        })
    }

    /// Backend family.
    #[must_use]
    pub const fn backend(&self) -> BackendIdentity {
        self.backend
    }

    /// Vendor-persistent device identifier.
    #[must_use]
    pub fn persistent_id(&self) -> &str {
        &self.persistent_id
    }

    /// Compute architecture.
    #[must_use]
    pub fn architecture(&self) -> &str {
        &self.architecture
    }

    /// Driver version used for measurements.
    #[must_use]
    pub const fn driver(&self) -> SoftwareVersion {
        self.driver
    }

    /// Stable, length-delimited digest of every identity field.
    #[must_use]
    pub fn digest(&self) -> u64 {
        Digest::new()
            .field(IDENTITY_SCHEMA)
            .field(b"device")
            .text(self.backend.name())
            .text(&self.persistent_id)
            .text(&self.architecture)
            .version(self.driver)
            .finish()
    }

    fn physical_key(&self) -> (BackendIdentity, &str) {
        (self.backend, &self.persistent_id)
    }
}

impl<D> Clone for DeviceFingerprint<D> {
    fn clone(&self) -> Self {
        Self {
            backend: self.backend,
            persistent_id: self.persistent_id.clone(),
            architecture: self.architecture.clone(),
            driver: self.driver,
            marker: PhantomData,
        }
    }
}

impl<D> fmt::Debug for DeviceFingerprint<D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeviceFingerprint")
            .field("backend", &self.backend)
            .field("persistent_id", &self.persistent_id)
            .field("architecture", &self.architecture)
            .field("driver", &self.driver)
            .finish()
    }
}

impl<D> PartialEq for DeviceFingerprint<D> {
    fn eq(&self, other: &Self) -> bool {
        self.backend == other.backend
            && self.persistent_id == other.persistent_id
            && self.architecture == other.architecture
            && self.driver == other.driver
    }
}

impl<D> Eq for DeviceFingerprint<D> {}

impl<D> PartialOrd for DeviceFingerprint<D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<D> Ord for DeviceFingerprint<D> {
    fn cmp(&self, other: &Self) -> Ordering {
        (
            self.backend,
            &self.persistent_id,
            &self.architecture,
            self.driver,
        )
            .cmp(&(
                other.backend,
                &other.persistent_id,
                &other.architecture,
                other.driver,
            ))
    }
}

impl<D> Hash for DeviceFingerprint<D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.backend.hash(state);
        self.persistent_id.hash(state);
        self.architecture.hash(state);
        self.driver.hash(state);
    }
}

/// Stable identity of the compiler which produced a tuned kernel.
pub struct CompilerFingerprint<D = Dyn> {
    backend: BackendIdentity,
    implementation: String,
    version: SoftwareVersion,
    target: String,
    options_digest: u64,
    marker: PhantomData<fn() -> D>,
}

impl<D: StaticBackend> CompilerFingerprint<D> {
    /// Constructs a compiler identity for a statically known backend.
    pub fn new(
        implementation: &str,
        version: SoftwareVersion,
        target: &str,
        semantic_options: &[&str],
    ) -> core::result::Result<Self, IdentityError> {
        Self::from_parts(
            D::BACKEND,
            implementation,
            version,
            target,
            semantic_options,
        )
    }

    /// Erases the static backend marker while retaining its runtime proof.
    #[must_use]
    pub fn erase(self) -> CompilerFingerprint<Dyn> {
        CompilerFingerprint {
            backend: self.backend,
            implementation: self.implementation,
            version: self.version,
            target: self.target,
            options_digest: self.options_digest,
            marker: PhantomData,
        }
    }
}

impl CompilerFingerprint<Dyn> {
    /// Constructs a runtime-selected compiler identity.
    pub fn new_dyn(
        backend: DeviceKind,
        implementation: &str,
        version: SoftwareVersion,
        target: &str,
        semantic_options: &[&str],
    ) -> core::result::Result<Self, IdentityError> {
        Self::from_parts(
            BackendIdentity::from_device_kind(backend)?,
            implementation,
            version,
            target,
            semantic_options,
        )
    }

    /// Projects a runtime identity to a statically known backend.
    pub fn try_into_static<D: StaticBackend>(
        self,
    ) -> core::result::Result<CompilerFingerprint<D>, IdentityError> {
        if self.backend != D::BACKEND {
            return Err(IdentityError::BackendMismatch {
                expected: D::BACKEND.name(),
                actual: self.backend.name(),
            });
        }
        Ok(CompilerFingerprint {
            backend: self.backend,
            implementation: self.implementation,
            version: self.version,
            target: self.target,
            options_digest: self.options_digest,
            marker: PhantomData,
        })
    }
}

impl<D> CompilerFingerprint<D> {
    fn from_parts(
        backend: BackendIdentity,
        implementation: &str,
        version: SoftwareVersion,
        target: &str,
        semantic_options: &[&str],
    ) -> core::result::Result<Self, IdentityError> {
        let implementation = checked_field("compiler_implementation", implementation)?;
        let target = checked_field("compiler_target", target)?;
        let mut options = Digest::new().field(b"incin.compiler.options.v1");
        for option in semantic_options {
            options = options.text(&checked_field("compiler_option", option)?);
        }
        Ok(Self {
            backend,
            implementation,
            version,
            target,
            options_digest: options.finish(),
            marker: PhantomData,
        })
    }

    /// Backend family compiled for.
    #[must_use]
    pub const fn backend(&self) -> BackendIdentity {
        self.backend
    }

    /// Compiler implementation, such as `nvrtc`.
    #[must_use]
    pub fn implementation(&self) -> &str {
        &self.implementation
    }

    /// Compiler version.
    #[must_use]
    pub const fn version(&self) -> SoftwareVersion {
        self.version
    }

    /// Stable compiler target, such as `sm_90`.
    #[must_use]
    pub fn target(&self) -> &str {
        &self.target
    }

    /// Length-delimited digest of semantic compiler options in application
    /// order.
    #[must_use]
    pub const fn options_digest(&self) -> u64 {
        self.options_digest
    }

    /// Stable digest of the full compiler identity.
    #[must_use]
    pub fn digest(&self) -> u64 {
        Digest::new()
            .field(IDENTITY_SCHEMA)
            .field(b"compiler")
            .text(self.backend.name())
            .text(&self.implementation)
            .version(self.version)
            .text(&self.target)
            .number(self.options_digest)
            .finish()
    }
}

impl<D> Clone for CompilerFingerprint<D> {
    fn clone(&self) -> Self {
        Self {
            backend: self.backend,
            implementation: self.implementation.clone(),
            version: self.version,
            target: self.target.clone(),
            options_digest: self.options_digest,
            marker: PhantomData,
        }
    }
}

impl<D> fmt::Debug for CompilerFingerprint<D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CompilerFingerprint")
            .field("backend", &self.backend)
            .field("implementation", &self.implementation)
            .field("version", &self.version)
            .field("target", &self.target)
            .field("options_digest", &self.options_digest)
            .finish()
    }
}

impl<D> PartialEq for CompilerFingerprint<D> {
    fn eq(&self, other: &Self) -> bool {
        self.backend == other.backend
            && self.implementation == other.implementation
            && self.version == other.version
            && self.target == other.target
            && self.options_digest == other.options_digest
    }
}

impl<D> Eq for CompilerFingerprint<D> {}

impl<D> PartialOrd for CompilerFingerprint<D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<D> Ord for CompilerFingerprint<D> {
    fn cmp(&self, other: &Self) -> Ordering {
        (
            self.backend,
            &self.implementation,
            self.version,
            &self.target,
            self.options_digest,
        )
            .cmp(&(
                other.backend,
                &other.implementation,
                other.version,
                &other.target,
                other.options_digest,
            ))
    }
}

impl<D> Hash for CompilerFingerprint<D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.backend.hash(state);
        self.implementation.hash(state);
        self.version.hash(state);
        self.target.hash(state);
        self.options_digest.hash(state);
    }
}

/// Device and compiler identity used by a local-kernel tuning key.
pub struct TuningEnvironmentFingerprint<D = Dyn> {
    device: DeviceFingerprint<D>,
    compiler: CompilerFingerprint<D>,
}

impl<D> TuningEnvironmentFingerprint<D> {
    fn checked(
        device: DeviceFingerprint<D>,
        compiler: CompilerFingerprint<D>,
    ) -> core::result::Result<Self, IdentityError> {
        if device.backend != compiler.backend {
            return Err(IdentityError::BackendMismatch {
                expected: device.backend.name(),
                actual: compiler.backend.name(),
            });
        }
        Ok(Self { device, compiler })
    }

    /// Stable device identity.
    #[must_use]
    pub const fn device(&self) -> &DeviceFingerprint<D> {
        &self.device
    }

    /// Stable compiler identity.
    #[must_use]
    pub const fn compiler(&self) -> &CompilerFingerprint<D> {
        &self.compiler
    }

    /// Stable digest of both identities.
    #[must_use]
    pub fn digest(&self) -> u64 {
        Digest::new()
            .field(IDENTITY_SCHEMA)
            .field(b"environment")
            .number(self.device.digest())
            .number(self.compiler.digest())
            .finish()
    }
}

impl<D: StaticBackend> TuningEnvironmentFingerprint<D> {
    /// Combines matching statically typed device and compiler identities.
    pub fn new(
        device: DeviceFingerprint<D>,
        compiler: CompilerFingerprint<D>,
    ) -> core::result::Result<Self, IdentityError> {
        Self::checked(device, compiler)
    }

    /// Erases the static backend marker while retaining its runtime proof.
    #[must_use]
    pub fn erase(self) -> TuningEnvironmentFingerprint<Dyn> {
        TuningEnvironmentFingerprint {
            device: self.device.erase(),
            compiler: self.compiler.erase(),
        }
    }
}

impl TuningEnvironmentFingerprint<Dyn> {
    /// Combines runtime identities, rejecting a device/compiler backend
    /// mismatch.
    pub fn new_dyn(
        device: DeviceFingerprint<Dyn>,
        compiler: CompilerFingerprint<Dyn>,
    ) -> core::result::Result<Self, IdentityError> {
        Self::checked(device, compiler)
    }

    /// Projects a runtime environment to a statically known backend.
    pub fn try_into_static<D: StaticBackend>(
        self,
    ) -> core::result::Result<TuningEnvironmentFingerprint<D>, IdentityError> {
        TuningEnvironmentFingerprint::new(
            self.device.try_into_static::<D>()?,
            self.compiler.try_into_static::<D>()?,
        )
    }
}

impl<D> Clone for TuningEnvironmentFingerprint<D> {
    fn clone(&self) -> Self {
        Self {
            device: self.device.clone(),
            compiler: self.compiler.clone(),
        }
    }
}

impl<D> fmt::Debug for TuningEnvironmentFingerprint<D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TuningEnvironmentFingerprint")
            .field("device", &self.device)
            .field("compiler", &self.compiler)
            .finish()
    }
}

impl<D> PartialEq for TuningEnvironmentFingerprint<D> {
    fn eq(&self, other: &Self) -> bool {
        self.device == other.device && self.compiler == other.compiler
    }
}

impl<D> Eq for TuningEnvironmentFingerprint<D> {}

impl<D> PartialOrd for TuningEnvironmentFingerprint<D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<D> Ord for TuningEnvironmentFingerprint<D> {
    fn cmp(&self, other: &Self) -> Ordering {
        (&self.device, &self.compiler).cmp(&(&other.device, &other.compiler))
    }
}

impl<D> Hash for TuningEnvironmentFingerprint<D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.device.hash(state);
        self.compiler.hash(state);
    }
}

/// A type-level, nonzero topology world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaticWorld<N: Unsigned + NonZero>(PhantomData<N>);

/// A directed communication link included in a topology identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TopologyLink {
    from: usize,
    to: usize,
    class: String,
}

impl TopologyLink {
    /// Records a directed link class.
    pub fn new(from: usize, to: usize, class: &str) -> core::result::Result<Self, IdentityError> {
        Ok(Self {
            from,
            to,
            class: checked_field("link_class", class)?,
        })
    }

    /// Source rank.
    #[must_use]
    pub const fn from(&self) -> usize {
        self.from
    }

    /// Destination rank.
    #[must_use]
    pub const fn to(&self) -> usize {
        self.to
    }

    /// Stable link class.
    #[must_use]
    pub fn class(&self) -> &str {
        &self.class
    }
}

/// Communication-library identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TransportFingerprint {
    library: String,
    version: SoftwareVersion,
}

impl TransportFingerprint {
    /// Constructs a transport identity.
    pub fn new(
        library: &str,
        version: SoftwareVersion,
    ) -> core::result::Result<Self, IdentityError> {
        Ok(Self {
            library: checked_field("transport_library", library)?,
            version,
        })
    }

    /// Library name.
    #[must_use]
    pub fn library(&self) -> &str {
        &self.library
    }

    /// Library version.
    #[must_use]
    pub const fn version(&self) -> SoftwareVersion {
        self.version
    }
}

/// Rank-to-process layout, excluding the observing process's local rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProcessLayoutFingerprint {
    processes: usize,
    ranks_per_process: usize,
}

impl ProcessLayoutFingerprint {
    /// Constructs a uniform process layout.
    #[must_use]
    pub const fn new(processes: usize, ranks_per_process: usize) -> Self {
        Self {
            processes,
            ranks_per_process,
        }
    }

    /// Number of processes.
    #[must_use]
    pub const fn processes(self) -> usize {
        self.processes
    }

    /// Number of ranks driven by each process.
    #[must_use]
    pub const fn ranks_per_process(self) -> usize {
        self.ranks_per_process
    }
}

/// Stable rank mapping, links, transport, and process layout.
///
/// `W = StaticWorld<N>` carries a nonzero world at compile time. `W = Dyn`
/// stores a runtime world and applies the same cardinality and alias checks.
pub struct TuningTopologyFingerprint<W = Dyn> {
    world: usize,
    devices: Vec<DeviceFingerprint<Dyn>>,
    links: Vec<TopologyLink>,
    transport: TransportFingerprint,
    layout: ProcessLayoutFingerprint,
    marker: PhantomData<fn() -> W>,
}

impl<N> TuningTopologyFingerprint<StaticWorld<N>>
where
    N: Unsigned + NonZero + 'static,
{
    /// Constructs a topology whose world is carried by `N`.
    pub fn new(
        devices: Vec<DeviceFingerprint<Dyn>>,
        links: Vec<TopologyLink>,
        transport: TransportFingerprint,
        layout: ProcessLayoutFingerprint,
    ) -> core::result::Result<Self, IdentityError> {
        Self::from_parts(N::USIZE, devices, links, transport, layout)
    }
}

impl TuningTopologyFingerprint<Dyn> {
    /// Constructs a runtime-selected topology.
    pub fn new_dyn(
        world: usize,
        devices: Vec<DeviceFingerprint<Dyn>>,
        links: Vec<TopologyLink>,
        transport: TransportFingerprint,
        layout: ProcessLayoutFingerprint,
    ) -> core::result::Result<Self, IdentityError> {
        Self::from_parts(world, devices, links, transport, layout)
    }

    /// Projects a runtime topology to a statically known nonzero world.
    pub fn try_into_static<N>(
        self,
    ) -> core::result::Result<TuningTopologyFingerprint<StaticWorld<N>>, IdentityError>
    where
        N: Unsigned + NonZero + 'static,
    {
        if self.world != N::USIZE {
            return Err(IdentityError::StaticWorldMismatch {
                expected: N::USIZE,
                actual: self.world,
            });
        }
        Ok(TuningTopologyFingerprint {
            world: self.world,
            devices: self.devices,
            links: self.links,
            transport: self.transport,
            layout: self.layout,
            marker: PhantomData,
        })
    }
}

impl<W> TuningTopologyFingerprint<W> {
    /// Erases a static or dynamic world marker while retaining the checked
    /// runtime value.
    #[must_use]
    pub fn erase(self) -> TuningTopologyFingerprint<Dyn> {
        TuningTopologyFingerprint {
            world: self.world,
            devices: self.devices,
            links: self.links,
            transport: self.transport,
            layout: self.layout,
            marker: PhantomData,
        }
    }

    fn from_parts(
        world: usize,
        devices: Vec<DeviceFingerprint<Dyn>>,
        mut links: Vec<TopologyLink>,
        transport: TransportFingerprint,
        layout: ProcessLayoutFingerprint,
    ) -> core::result::Result<Self, IdentityError> {
        if world == 0 {
            return Err(IdentityError::ZeroWorld);
        }
        if devices.len() != world {
            return Err(IdentityError::WorldMismatch {
                world,
                devices: devices.len(),
            });
        }
        for second_rank in 0..devices.len() {
            if let Some(first_rank) = (0..second_rank).find(|&first_rank| {
                devices[first_rank].physical_key() == devices[second_rank].physical_key()
            }) {
                return Err(IdentityError::AliasedDevice {
                    persistent_id: devices[second_rank].persistent_id.clone(),
                    first_rank,
                    second_rank,
                });
            }
        }
        links.sort();
        for (index, link) in links.iter().enumerate() {
            if link.from >= world || link.to >= world {
                return Err(IdentityError::LinkOutOfRange {
                    from: link.from,
                    to: link.to,
                    world,
                });
            }
            if link.from == link.to {
                return Err(IdentityError::SelfLink { rank: link.from });
            }
            if index > 0 && links[index - 1].from == link.from && links[index - 1].to == link.to {
                return Err(IdentityError::DuplicateLink {
                    from: link.from,
                    to: link.to,
                });
            }
        }
        let covered = layout
            .processes
            .checked_mul(layout.ranks_per_process)
            .filter(|&covered| covered == world);
        if covered.is_none() {
            return Err(IdentityError::ProcessLayoutMismatch {
                processes: layout.processes,
                ranks_per_process: layout.ranks_per_process,
                world,
            });
        }
        Ok(Self {
            world,
            devices,
            links,
            transport,
            layout,
            marker: PhantomData,
        })
    }

    /// Number of ranks in the topology.
    #[must_use]
    pub const fn world(&self) -> usize {
        self.world
    }

    /// Stable identity bound to each rank, in rank order.
    #[must_use]
    pub fn devices(&self) -> &[DeviceFingerprint<Dyn>] {
        &self.devices
    }

    /// Directed links in canonical order.
    #[must_use]
    pub fn links(&self) -> &[TopologyLink] {
        &self.links
    }

    /// Communication-library identity.
    #[must_use]
    pub const fn transport(&self) -> &TransportFingerprint {
        &self.transport
    }

    /// Rank-to-process layout.
    #[must_use]
    pub const fn layout(&self) -> ProcessLayoutFingerprint {
        self.layout
    }

    /// Stable digest excluding process-local ordinals and observing rank.
    #[must_use]
    pub fn digest(&self) -> u64 {
        let mut digest = Digest::new()
            .field(IDENTITY_SCHEMA)
            .field(b"topology")
            .number(self.world as u64);
        for device in &self.devices {
            digest = digest.number(device.digest());
        }
        for link in &self.links {
            digest = digest
                .number(link.from as u64)
                .number(link.to as u64)
                .text(&link.class);
        }
        digest
            .text(&self.transport.library)
            .version(self.transport.version)
            .number(self.layout.processes as u64)
            .number(self.layout.ranks_per_process as u64)
            .finish()
    }
}

impl<W> Clone for TuningTopologyFingerprint<W> {
    fn clone(&self) -> Self {
        Self {
            world: self.world,
            devices: self.devices.clone(),
            links: self.links.clone(),
            transport: self.transport.clone(),
            layout: self.layout,
            marker: PhantomData,
        }
    }
}

impl<W> fmt::Debug for TuningTopologyFingerprint<W> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TuningTopologyFingerprint")
            .field("world", &self.world)
            .field("devices", &self.devices)
            .field("links", &self.links)
            .field("transport", &self.transport)
            .field("layout", &self.layout)
            .finish()
    }
}

impl<W> PartialEq for TuningTopologyFingerprint<W> {
    fn eq(&self, other: &Self) -> bool {
        self.world == other.world
            && self.devices == other.devices
            && self.links == other.links
            && self.transport == other.transport
            && self.layout == other.layout
    }
}

impl<W> Eq for TuningTopologyFingerprint<W> {}

#[cfg(feature = "cuda")]
impl DeviceFingerprint<Cuda> {
    /// Queries UUID, architecture, and driver version from a live CUDA
    /// context. The context ordinal is used only in diagnostics and never
    /// enters the returned identity.
    pub fn from_cuda_context(
        context: &cudarc::driver::CudaContext,
    ) -> core::result::Result<Self, IdentityError> {
        let ordinal = context.ordinal();
        let uuid = context.uuid().map_err(|error| IdentityError::CudaQuery {
            component: "device UUID",
            message: format!("device {ordinal}: {error:?}"),
        })?;
        let (major, minor) =
            context
                .compute_capability()
                .map_err(|error| IdentityError::CudaQuery {
                    component: "compute capability",
                    message: format!("device {ordinal}: {error:?}"),
                })?;
        if major < 0 || minor < 0 {
            return Err(IdentityError::CudaQuery {
                component: "compute capability",
                message: format!("device {ordinal} returned sm_{major}{minor}"),
            });
        }
        Self::new(
            &format_cuda_uuid(uuid.bytes),
            &format!("sm_{major}{minor}"),
            cuda_driver_version()?,
        )
    }
}

#[cfg(feature = "cuda")]
impl CompilerFingerprint<Cuda> {
    /// Queries NVRTC and the target architecture used for this CUDA context.
    pub fn from_cuda_context(
        context: &cudarc::driver::CudaContext,
    ) -> core::result::Result<Self, IdentityError> {
        let ordinal = context.ordinal();
        let (major, minor) =
            context
                .compute_capability()
                .map_err(|error| IdentityError::CudaQuery {
                    component: "compiler target",
                    message: format!("device {ordinal}: {error:?}"),
                })?;
        if major < 0 || minor < 0 {
            return Err(IdentityError::CudaQuery {
                component: "compiler target",
                message: format!("device {ordinal} returned sm_{major}{minor}"),
            });
        }
        Self::new(
            "nvrtc",
            nvrtc_version()?,
            &format!("sm_{major}{minor}"),
            &[
                "incin-nvrtc-options-v1",
                "default-math",
                "cuda-include-discovery-v1",
            ],
        )
    }
}

#[cfg(feature = "cuda")]
impl TuningEnvironmentFingerprint<Cuda> {
    /// Queries the complete CUDA device/compiler environment.
    pub fn from_cuda_context(
        context: &cudarc::driver::CudaContext,
    ) -> incin_core::prelude::Result<Self> {
        use incin_core::prelude::Error;
        Self::new(
            DeviceFingerprint::from_cuda_context(context)
                .map_err(|error| Error::Msg(error.to_string()))?,
            CompilerFingerprint::from_cuda_context(context)
                .map_err(|error| Error::Msg(error.to_string()))?,
        )
        .map_err(|error| Error::Msg(error.to_string()))
    }
}

#[cfg(feature = "cuda")]
fn cuda_driver_version() -> core::result::Result<SoftwareVersion, IdentityError> {
    let mut encoded = 0_i32;
    let result = unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut encoded) };
    if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(IdentityError::CudaQuery {
            component: "driver version",
            message: format!("{result:?}"),
        });
    }
    if encoded < 0 {
        return Err(IdentityError::CudaQuery {
            component: "driver version",
            message: format!("negative encoded version {encoded}"),
        });
    }
    let encoded = encoded as u32;
    Ok(SoftwareVersion::new(
        encoded / 1000,
        (encoded % 1000) / 10,
        encoded % 10,
    ))
}

#[cfg(feature = "cuda")]
fn nvrtc_version() -> core::result::Result<SoftwareVersion, IdentityError> {
    let mut major = 0_i32;
    let mut minor = 0_i32;
    let result = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
    if result != cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS {
        return Err(IdentityError::CudaQuery {
            component: "NVRTC version",
            message: format!("{result:?}"),
        });
    }
    if major < 0 || minor < 0 {
        return Err(IdentityError::CudaQuery {
            component: "NVRTC version",
            message: format!("negative version {major}.{minor}"),
        });
    }
    Ok(SoftwareVersion::new(major as u32, minor as u32, 0))
}

#[allow(dead_code)]
fn format_cuda_uuid(bytes: [core::ffi::c_char; 16]) -> String {
    let mut output = String::from("GPU-");
    for (index, byte) in bytes.into_iter().enumerate() {
        if matches!(index, 4 | 6 | 8 | 10) {
            output.push('-');
        }
        use core::fmt::Write as _;
        write!(&mut output, "{:02x}", byte as u8).expect("writing to String cannot fail");
    }
    output
}

#[derive(Clone, Copy)]
struct Digest(u64);

impl Digest {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    const fn new() -> Self {
        Self(Self::OFFSET)
    }

    fn bytes(mut self, bytes: &[u8]) -> Self {
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(Self::PRIME);
        }
        self
    }

    fn field(self, bytes: &[u8]) -> Self {
        self.number(bytes.len() as u64).bytes(bytes)
    }

    fn text(self, value: &str) -> Self {
        self.field(value.as_bytes())
    }

    fn number(self, value: u64) -> Self {
        self.bytes(&value.to_le_bytes())
    }

    fn version(self, version: SoftwareVersion) -> Self {
        self.number(u64::from(version.major))
            .number(u64::from(version.minor))
            .number(u64::from(version.patch))
    }

    const fn finish(self) -> u64 {
        self.0
    }
}
