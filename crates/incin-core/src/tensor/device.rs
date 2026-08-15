//! Device types for the incin tensor system.
//!
//! There are three tiers of device specification, ordered from most
//! dynamic to most static:
//!
//! ## Tier 1 — Fully Runtime (`Dyn`)
//!
//! Neither the backend family nor the device ordinal is known at compile
//! time. The user passes a [`DeviceId`] at construction time and everything
//! is dispatched at runtime.
//!
//! ```text
//! Tensor::<Dyn, IncinBackend<Dyn>>::zeros(([2, 3], DTypeId::F32, DeviceId::cuda(1)))
//! ```
//!
//! ## Tier 2 — Partial Compile-Time (`Cuda` / `Wgpu`)
//!
//! The backend family (CUDA or WGPU) is known at compile time, but the
//! specific device ordinal (which GPU to use) is provided through an explicit
//! selector value. Useful when you know you want CUDA but need to select the GPU
//! based on e.g. command-line flags.
//!
//! ```text
//! Tensor::<s![2, 3], IncinBackend<Cuda>>::zeros(((), Cuda::new(2)))
//! ```
//!
//! The argument is a 2-tuple, not the bare selector. A fully static shape's
//! `Shape::Arg` is a tuple of units (`((), ())` here), and `arg_into`'s
//! `NotUnit` marker counts that as an argument the caller supplied, so the
//! shape slot is already occupied. The leading `()` fills it and the selector
//! lands in the device slot. Passing `Cuda::new(2)` alone instead makes
//! `ArgInto` try to read it as the *shape*, and the mismatch surfaces as an
//! unsatisfied `ArgInto<TensorArgsData<..>>` bound rather than as anything
//! that names the device.
//!
//! ## Tier 3 — Fully Static Selection (`CudaN<N>` / `WgpuN<N>`)
//!
//! Both the backend family and the device ordinal are encoded at the type
//! level via [`typenum`] unsigned integers. The tensor type fully describes
//! the requested logical device address. No constructor argument is required.
//! Hardware existence, driver compatibility, and capabilities are necessarily
//! validated when the program initializes that device at runtime.
//!
//! ```text
//! Tensor::<s![2, 3], IncinBackend<CudaN<U1>>>::zeros(())  // always GPU 1
//! ```

use core::fmt::Debug;
use core::marker::PhantomData;

use crate::err::Result;
use crate::shapes::Dyn;

/// A type-level compute device.
///
/// | Type             | Backend at compile time | Ordinal at compile time | Constructor arg |
/// |------------------|------------------------|-------------------------|-----------------|
/// | `Dyn`            | ✗                      | ✗                       | `DeviceId`      |
/// | `Cuda` / `Wgpu`  | ✓                      | ✗                       | selector value  |
/// | `CudaN<N>` / `WgpuN<N>` | ✓               | ✓                       | `()`            |
pub trait Device: 'static + Send + Sync + Clone + Eq + PartialEq + Debug + Sized {
    /// The user-facing constructor argument:
    /// - `DeviceId` for `Dyn` (fully runtime)
    /// - an explicit selector for `Cuda`/`Wgpu` (partial — ordinal at runtime)
    /// - `()` for `CudaN<N>`/`WgpuN<N>` (fully static)
    type Arg: Clone;
    /// The runtime-stored representation:
    /// - `DeviceId` for `Dyn`
    /// - the selector for `Cuda`/`Wgpu`
    /// - `PhantomData<Self>` for `CudaN<N>`/`WgpuN<N>`
    type Field: Debug + Clone + Default;
    /// Converts a user-facing `Arg` into the stored `Field` representation.
    fn init(arg: Self::Arg) -> Self::Field;
    /// Resolves this device's runtime [`DeviceId`].
    fn to_incin(dev: &Self::Field) -> Result<DeviceId>;
}

/// A [`Device`] whose logical selector is **fully known at compile time** —
/// both the backend family and ordinal are encoded in the type. This does not
/// prove that matching hardware exists on the runtime host. Takes no
/// constructor argument (`Arg = ()`).
///
/// Implemented by `Cpu`, `CudaN<N: Unsigned>`, and `WgpuN<N: Unsigned>`.
pub trait ConstDevice: Default + Device<Arg = ()> {}

// ============================================================================
// Tier 1: Fully Runtime — Dyn
// ============================================================================

impl Device for Dyn {
    /// The runtime-chosen device — user passes a full [`DeviceId`].
    type Arg = DeviceId;
    /// Stored directly — `Dyn`'s whole point is deferring device choice
    /// to runtime, so `Field` is just the `DeviceId` itself.
    type Field = DeviceId;

    /// Already a `DeviceId` — returned as-is.
    fn to_incin(dev: &Self::Field) -> Result<DeviceId> {
        Ok(*dev)
    }

    /// Stores the `DeviceId` verbatim.
    fn init(arg: Self::Arg) -> Self::Field {
        arg
    }
}

// ============================================================================
// Tier 2: Partial Compile-Time — Cuda / Wgpu (runtime ordinal)
// ============================================================================

#[cfg(feature = "cuda")]
mod cuda_partial {
    use super::{Device, DeviceId, Result};

    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
    /// **Tier 2** CUDA device: backend kind known at compile time, ordinal
    /// supplied at runtime as a `usize`.
    ///
    /// Use this when you know you want CUDA but which GPU to use is
    /// determined at runtime (e.g. via a CLI flag or environment variable).
    ///
    /// For a fully static device selector use `CudaN<N>` where `N` is a
    /// [`typenum`] unsigned (e.g. `Cuda<U0>` for GPU 0).
    pub struct Cuda {
        ordinal: usize,
    }

    impl Cuda {
        /// Selects a logical CUDA ordinal. Hardware availability is checked
        /// only when a backend initializes the selector.
        #[must_use]
        pub const fn new(ordinal: usize) -> Self {
            Self { ordinal }
        }

        /// Returns the selected logical ordinal.
        #[must_use]
        pub const fn ordinal(self) -> usize {
            self.ordinal
        }
    }

    impl Device for Cuda {
        /// Ordinal supplied at construction time.
        type Arg = Self;
        /// Validated selector stored directly.
        type Field = Self;

        fn to_incin(dev: &Self::Field) -> Result<DeviceId> {
            Ok(DeviceId::cuda(dev.ordinal))
        }

        fn init(arg: Self::Arg) -> Self::Field {
            arg
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_partial::Cuda;

#[cfg(feature = "wgpu")]
mod wgpu_partial {
    use super::{Device, DeviceId, Result};

    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
    /// **Tier 2** WGPU device: backend kind known at compile time, ordinal
    /// supplied at runtime as a `usize`.
    ///
    /// Use this when you know you want WGPU but which adapter to use is
    /// determined at runtime.
    ///
    /// For a fully static device selector use `WgpuN<N>` where `N` is a
    /// [`typenum`] unsigned (e.g. `Wgpu<U0>` for adapter 0).
    pub struct Wgpu {
        ordinal: usize,
    }

    impl Wgpu {
        /// Selects a logical WGPU adapter ordinal without claiming that the
        /// adapter exists on the current host.
        #[must_use]
        pub const fn new(ordinal: usize) -> Self {
            Self { ordinal }
        }

        /// Returns the selected logical ordinal.
        #[must_use]
        pub const fn ordinal(self) -> usize {
            self.ordinal
        }
    }

    impl Device for Wgpu {
        /// Ordinal supplied at construction time.
        type Arg = Self;
        /// Validated selector stored directly.
        type Field = Self;

        fn to_incin(dev: &Self::Field) -> Result<DeviceId> {
            Ok(DeviceId::wgpu(dev.ordinal))
        }

        fn init(arg: Self::Arg) -> Self::Field {
            arg
        }
    }
}

#[cfg(feature = "wgpu")]
pub use wgpu_partial::Wgpu;

// ============================================================================
// Tier 3: Fully Static Selection — CudaN<N> / WgpuN<N> (typenum ordinal)
// ============================================================================

#[cfg(feature = "cuda")]
mod cuda_static {
    use super::{ConstDevice, Device, DeviceId, PhantomData, Result};
    use typenum::{U0, Unsigned};

    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
    /// **Tier 3** CUDA device: both the backend kind *and* the device
    /// ordinal `N` are fully known at compile time via [`typenum`].
    ///
    /// This is a zero-sized type — no runtime data is stored. The default
    /// ordinal is `U0` (GPU 0).
    ///
    /// ```text
    /// // Always on GPU 0 — no runtime arg required
    /// Tensor::<s![2, 3], IncinBackend<CudaN<U0>>>::zeros(())
    ///
    /// // Always on GPU 2
    /// Tensor::<s![2, 3], IncinBackend<CudaN<U2>>>::zeros(())
    /// ```
    pub struct CudaN<N: Unsigned = U0>(PhantomData<N>);

    impl<N: Unsigned + 'static + Send + Sync + Clone + Eq + PartialEq + Debug> ConstDevice
        for CudaN<N>
    {
    }

    use core::fmt::Debug;

    impl<N: Unsigned + 'static + Send + Sync + Clone + Eq + PartialEq + Debug> Device for CudaN<N> {
        /// No constructor argument — the device ordinal `N` is compile-time-fixed.
        type Arg = ();
        /// Zero-sized: `N` alone identifies the device.
        type Field = PhantomData<Self>;

        /// Resolves to CUDA device ordinal `N::USIZE`.
        fn to_incin(_: &Self::Field) -> Result<DeviceId> {
            Ok(DeviceId::cuda(N::USIZE))
        }

        /// No-op: nothing to convert.
        fn init(_: Self::Arg) -> Self::Field {
            PhantomData
        }
    }
}

#[cfg(feature = "cuda")]
pub use cuda_static::CudaN;

#[cfg(feature = "wgpu")]
mod wgpu_static {
    use super::{ConstDevice, Device, DeviceId, PhantomData, Result};
    use typenum::{U0, Unsigned};

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    /// **Tier 3** WGPU device: both the backend kind *and* the adapter
    /// ordinal `N` are fully known at compile time via [`typenum`].
    ///
    /// This is a zero-sized type — no runtime data is stored. The default
    /// ordinal is `U0` (adapter 0).
    ///
    /// ```text
    /// // Always on adapter 0 — no runtime arg required
    /// Tensor::<s![2, 3], IncinBackend<WgpuN<U0>>>::zeros(())
    /// ```
    pub struct WgpuN<N: Unsigned = U0>(PhantomData<N>);

    impl<N: Unsigned + 'static + Send + Sync + Clone + Eq + PartialEq + Debug> ConstDevice
        for WgpuN<N>
    {
    }

    use core::fmt::Debug;

    impl<N: Unsigned + 'static + Send + Sync + Clone + Eq + PartialEq + Debug> Device for WgpuN<N> {
        /// No constructor argument — the adapter ordinal `N` is compile-time-fixed.
        type Arg = ();
        /// Zero-sized: `N` alone identifies the device.
        type Field = PhantomData<Self>;

        /// Resolves to WGPU adapter ordinal `N::USIZE`.
        fn to_incin(_: &Self::Field) -> Result<DeviceId> {
            Ok(DeviceId::wgpu(N::USIZE))
        }

        /// No-op: nothing to convert.
        fn init(_: Self::Arg) -> Self::Field {
            PhantomData
        }
    }
}

#[cfg(feature = "wgpu")]
pub use wgpu_static::WgpuN;

#[cfg(feature = "metal")]
mod metal_partial {
    use super::{Device, DeviceId, Result};

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    /// **Tier 2** Metal device: backend kind known at compile time, ordinal
    /// supplied at runtime as a `usize`.
    pub struct Metal {
        ordinal: usize,
    }

    impl Metal {
        /// Selects a logical Metal device ordinal without claiming that the
        /// device exists on the current host.
        #[must_use]
        pub const fn new(ordinal: usize) -> Self {
            Self { ordinal }
        }

        /// Returns the selected logical ordinal.
        #[must_use]
        pub const fn ordinal(&self) -> usize {
            self.ordinal
        }
    }

    impl Device for Metal {
        type Arg = Self;
        type Field = Self;

        fn to_incin(dev: &Self::Field) -> Result<DeviceId> {
            Ok(DeviceId::metal(dev.ordinal))
        }

        fn init(arg: Self::Arg) -> Self::Field {
            arg
        }
    }
}

#[cfg(feature = "metal")]
pub use metal_partial::Metal;

#[cfg(feature = "metal")]
mod metal_static {
    use super::{ConstDevice, Device, DeviceId, PhantomData, Result};
    use core::fmt::Debug;
    use typenum::{U0, Unsigned};

    #[derive(Debug, Default, Clone, PartialEq, Eq)]
    /// **Tier 3** Metal device: both backend kind and device ordinal `N` are known at compile time.
    pub struct MetalN<N: Unsigned = U0>(PhantomData<N>);

    impl<N: Unsigned + 'static + Send + Sync + Clone + Eq + PartialEq + Debug> ConstDevice
        for MetalN<N>
    {
    }

    impl<N: Unsigned + 'static + Send + Sync + Clone + Eq + PartialEq + Debug> Device for MetalN<N> {
        type Arg = ();
        type Field = PhantomData<Self>;

        fn to_incin(_: &Self::Field) -> Result<DeviceId> {
            Ok(DeviceId::metal(N::USIZE))
        }

        fn init(_: Self::Arg) -> Self::Field {
            PhantomData
        }
    }
}

#[cfg(feature = "metal")]
pub use metal_static::MetalN;

// ============================================================================
// CPU — always fully static (there is only one CPU)
// ============================================================================

#[derive(Debug, Default, Clone, PartialEq, Eq)]
/// The CPU device. A zero-sized type — there is only one CPU, so no ordinal
/// is needed. This is a **Tier 3** (fully static) device selector.
pub struct Cpu;

impl ConstDevice for Cpu {}

impl Device for Cpu {
    /// No constructor argument needed — there is only one CPU device.
    type Arg = ();
    /// Zero-sized: there is nothing to store.
    type Field = PhantomData<Self>;

    /// Always resolves to `DeviceId::cpu()`.
    fn to_incin(_: &Self::Field) -> Result<DeviceId> {
        Ok(DeviceId::cpu())
    }

    /// No-op: nothing to convert.
    fn init(_: Self::Arg) -> Self::Field {
        PhantomData
    }
}

// ============================================================================
// DeviceId and DeviceKind — runtime device identity
// ============================================================================

#[non_exhaustive]
#[derive(Debug, Clone, Copy, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
/// The runtime-identifiable backend family a `DeviceId` belongs to.
pub enum DeviceKind {
    /// The CPU backend family.
    Cpu,
    /// The CUDA backend family.
    Cuda,
    /// The WGPU backend family.
    Wgpu,
    /// The Metal backend family for Apple Silicon.
    Metal,
    /// An externally defined backend family identified by a stable namespace key.
    ///
    /// The key is owned by the external backend. Incin does not interpret it,
    /// so adding a backend does not require changing this enum.
    Custom(u64),
}

impl DeviceKind {
    /// The lowercase name used in diagnostics, generated documentation, and
    /// `cargo incin doctor`'s report.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Wgpu => "wgpu",
            Self::Metal => "metal",
            Self::Custom(_) => "custom",
        }
    }

    /// Returns the external namespace key, if this is a custom backend kind.
    #[must_use]
    pub const fn custom_key(self) -> Option<u64> {
        match self {
            Self::Custom(key) => Some(key),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, serde::Serialize, serde::Deserialize)]
/// A runtime device identifier: a backend family ([`DeviceKind`]) plus an
/// ordinal distinguishing multiple devices of the same family (e.g. GPU 0
/// vs. GPU 1). This is the [`Device`] trait's runtime counterpart — every
/// `Device::to_incin` resolves to one of these.
pub struct DeviceId {
    kind: DeviceKind,
    ordinal: usize,
}

impl Default for DeviceId {
    fn default() -> Self {
        Self::cpu()
    }
}

impl DeviceId {
    /// Returns the backend family.
    pub const fn kind(self) -> DeviceKind {
        self.kind
    }

    /// Returns the ordinal within the backend family.
    pub const fn ordinal(self) -> usize {
        self.ordinal
    }

    /// The single CPU device, usable in a `const` context.
    pub const CPU: Self = Self {
        kind: DeviceKind::Cpu,
        ordinal: 0,
    };

    /// The single CPU device (ordinal always 0).
    pub fn cpu() -> Self {
        Self {
            kind: DeviceKind::Cpu,
            ordinal: 0,
        }
    }

    /// A CUDA device at ordinal `ord`.
    pub fn cuda(ord: usize) -> Self {
        Self {
            kind: DeviceKind::Cuda,
            ordinal: ord,
        }
    }

    /// A WGPU device at ordinal `ord`.
    pub fn wgpu(ord: usize) -> Self {
        Self {
            kind: DeviceKind::Wgpu,
            ordinal: ord,
        }
    }

    /// A Metal device at ordinal `ord`.
    pub fn metal(ord: usize) -> Self {
        Self {
            kind: DeviceKind::Metal,
            ordinal: ord,
        }
    }

    /// An externally defined backend device at ordinal `ord`.
    ///
    /// `namespace` must be a stable key chosen by the external backend. The
    /// key is carried through metadata and serialization without requiring
    /// Incin to know the backend's type.
    pub const fn custom(namespace: u64, ord: usize) -> Self {
        Self {
            kind: DeviceKind::Custom(namespace),
            ordinal: ord,
        }
    }
}

/// Whether this build was compiled with the `cuda` feature enabled
/// (does not check for actual CUDA hardware/drivers at runtime).
pub const fn cuda_is_available() -> bool {
    cfg!(feature = "cuda")
}
/// Whether this build was compiled with the `wgpu` feature enabled
/// (does not check for an actual GPU adapter at runtime).
pub const fn wgpu_is_available() -> bool {
    cfg!(feature = "wgpu")
}
/// Whether this build was compiled with the `metal` feature enabled.
pub const fn metal_is_available() -> bool {
    cfg!(feature = "metal")
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
/// A Metal device index, used as a hashable/orderable key.
pub struct MetalDevice {
    id: usize,
}

impl MetalDevice {
    /// Creates a new instance with the given device ordinal.
    pub fn new(id: usize) -> Self {
        Self { id }
    }

    /// Returns the logical device ordinal.
    #[must_use]
    pub const fn ordinal(self) -> usize {
        self.id
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
/// A CUDA device index, used as a hashable/orderable key (e.g. for
/// per-device kernel caches) distinct from the type-level `Cuda`/`CudaN<N>` markers.
pub struct CudaDevice {
    id: usize,
}

impl CudaDevice {
    /// Creates a new instance with the given device ordinal.
    pub fn new(id: usize) -> Self {
        Self { id }
    }

    /// Returns the logical device ordinal.
    #[must_use]
    pub const fn ordinal(self) -> usize {
        self.id
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
/// A WGPU device index, used as a hashable/orderable key distinct from
/// the type-level `Wgpu`/`WgpuN<N>` markers.
pub struct WgpuDevice {
    id: usize,
}

impl WgpuDevice {
    /// Creates a new instance with the given device ordinal.
    pub fn new(id: usize) -> Self {
        Self { id }
    }

    /// Returns the logical device ordinal.
    #[must_use]
    pub const fn ordinal(self) -> usize {
        self.id
    }
}

/// An ordered, non-empty set of devices a run is asked to use.
///
/// `UX-001`. Appendix A.8 assigns this row `DeviceSet` and `DevicePreference`
/// alongside a `BackendKind`; that third type is [`DeviceKind`], which already
/// means "the runtime-identifiable backend family a `DeviceId` belongs to", so
/// this builds on it rather than adding a second spelling (`D-008`).
///
/// Order is kept because it is meaningful — the first device is where a
/// single-device run happens and where rank 0 sits — and duplicates are
/// rejected, because a set naming the same GPU twice describes a run that
/// cannot exist.
///
/// ```rust
/// use incin_core::tensor::device::{DeviceKind, DeviceSet};
///
/// let one = DeviceSet::cpu();
/// assert_eq!(one.len(), 1);
/// assert!(!one.is_multi_device());
///
/// let three = DeviceSet::cuda(0..3).unwrap();
/// assert_eq!(three.len(), 3);
/// assert!(three.is_multi_device());
/// assert_eq!(three.kind(), Some(DeviceKind::Cuda));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceSet {
    devices: alloc::vec::Vec<DeviceId>,
}

/// Why a [`DeviceSet`] could not be built.
///
/// Separate from `ShapeError` because none of these are about shapes, and
/// separate from the trainer's own error because a `DeviceSet` is constructed
/// long before there is a trainer to report through.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DeviceSetError {
    /// The set would have been empty.
    ///
    /// A run has to happen somewhere, and the alternative to rejecting this is
    /// picking a device on the caller's behalf — which is the silent fallback
    /// sec. 2 rules out.
    Empty,
    /// The same device was named more than once.
    Duplicate {
        /// The device that appeared twice.
        device: DeviceId,
    },
    /// The set mixes backend families.
    ///
    /// Rejected for now rather than forever: a CPU/GPU heterogeneous run is a
    /// real thing, but it is `DST-016`'s, and a set that silently permits it
    /// here would be validated by nothing.
    Mixed {
        /// The family the first device belongs to.
        first: DeviceKind,
        /// The family that disagreed with it.
        found: DeviceKind,
    },
}

impl core::fmt::Display for DeviceSetError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Empty => f.write_str("a device set must name at least one device"),
            Self::Duplicate { device } => write!(
                f,
                "device {}:{} appears more than once",
                device.kind().name(),
                device.ordinal()
            ),
            Self::Mixed { first, found } => write!(
                f,
                "a device set cannot mix backend families: {} and {}",
                first.name(),
                found.name()
            ),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DeviceSetError {}

impl DeviceSet {
    /// Builds a set from devices in the order given.
    ///
    /// # Errors
    ///
    /// [`DeviceSetError`] if the list is empty, repeats a device, or mixes
    /// backend families.
    pub fn new(
        devices: impl IntoIterator<Item = DeviceId>,
    ) -> core::result::Result<Self, DeviceSetError> {
        let devices: alloc::vec::Vec<DeviceId> = devices.into_iter().collect();
        let first = *devices.first().ok_or(DeviceSetError::Empty)?;
        for (index, &device) in devices.iter().enumerate() {
            if device.kind() != first.kind() {
                return Err(DeviceSetError::Mixed {
                    first: first.kind(),
                    found: device.kind(),
                });
            }
            if devices[..index].contains(&device) {
                return Err(DeviceSetError::Duplicate { device });
            }
        }
        Ok(Self { devices })
    }

    /// The single CPU device.
    #[must_use]
    pub fn cpu() -> Self {
        Self {
            devices: alloc::vec![DeviceId::cpu()],
        }
    }

    /// CUDA devices at the given ordinals — `DeviceSet::cuda(0..3)` in sec. 2's
    /// example.
    ///
    /// # Errors
    ///
    /// [`DeviceSetError::Empty`] if the range is empty. An ordinal range cannot
    /// repeat or mix families, so those variants are unreachable here.
    pub fn cuda(
        ordinals: impl IntoIterator<Item = usize>,
    ) -> core::result::Result<Self, DeviceSetError> {
        Self::new(ordinals.into_iter().map(DeviceId::cuda))
    }

    /// WGPU devices at the given ordinals.
    ///
    /// # Errors
    ///
    /// [`DeviceSetError::Empty`] if the range is empty.
    pub fn wgpu(
        ordinals: impl IntoIterator<Item = usize>,
    ) -> core::result::Result<Self, DeviceSetError> {
        Self::new(ordinals.into_iter().map(DeviceId::wgpu))
    }

    /// The devices, in the order they were given.
    #[must_use]
    pub fn devices(&self) -> &[DeviceId] {
        &self.devices
    }

    /// The device a single-device run happens on, and rank 0 otherwise.
    #[must_use]
    pub fn primary(&self) -> DeviceId {
        self.devices[0]
    }

    /// How many devices the set names. Never zero.
    #[must_use]
    pub fn len(&self) -> usize {
        self.devices.len()
    }

    /// Always `false`. Present because clippy asks for it beside `len`, and
    /// because a caller reading `is_empty` should get the real answer rather
    /// than have to know the invariant.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// Whether this set needs collectives to run at all.
    #[must_use]
    pub fn is_multi_device(&self) -> bool {
        self.devices.len() > 1
    }

    /// The one backend family every device in the set belongs to.
    ///
    /// `Option` only because [`DeviceKind`] is `#[non_exhaustive]`: the set is
    /// never empty, so this is `Some` for every family this build knows.
    #[must_use]
    pub fn kind(&self) -> Option<DeviceKind> {
        self.devices.first().map(|device| device.kind())
    }
}

/// What a caller wants when they have not named specific devices.
///
/// Distinct from [`DeviceSet`] on purpose: a preference is resolved against a
/// machine and can fail, a set is already resolved. Keeping them one type is
/// what makes "I asked for CUDA and got CPU" possible to express, and sec. 2
/// rules that out — "'easy' must not mean silent CPU transfer".
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum DevicePreference {
    /// Use exactly these devices, or fail.
    Exactly(DeviceSet),
    /// Use the fastest family this build was compiled with and that the
    /// machine actually has, falling back through the preference order.
    ///
    /// This is the one variant permitted to end up on the CPU, because it is
    /// the one where the caller said they did not mind.
    Fastest,
    /// Use the CPU.
    Cpu,
}

impl Default for DevicePreference {
    /// [`DevicePreference::Cpu`].
    ///
    /// Not `Fastest`: a default that silently moves a run onto a GPU when one
    /// appears is the same class of surprise as one that silently moves it off.
    fn default() -> Self {
        Self::Cpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_device_set_keeps_the_order_it_was_given() {
        let set = DeviceSet::new([DeviceId::cuda(2), DeviceId::cuda(0), DeviceId::cuda(1)])
            .expect("three distinct CUDA devices are a valid set");
        assert_eq!(set.primary(), DeviceId::cuda(2));
        assert_eq!(set.len(), 3);
        assert!(set.is_multi_device());
    }

    #[test]
    fn a_device_set_rejects_what_cannot_describe_a_run() {
        assert_eq!(DeviceSet::new([]), Err(DeviceSetError::Empty));
        assert_eq!(
            DeviceSet::new([DeviceId::cuda(0), DeviceId::cuda(0)]),
            Err(DeviceSetError::Duplicate {
                device: DeviceId::cuda(0)
            })
        );
        assert_eq!(
            DeviceSet::new([DeviceId::cuda(0), DeviceId::cpu()]),
            Err(DeviceSetError::Mixed {
                first: DeviceKind::Cuda,
                found: DeviceKind::Cpu
            })
        );
    }

    /// `DeviceSet::cuda(0..3)` is the literal call in sec. 2's example.
    #[test]
    fn the_rfcs_three_gpu_call_builds_three_cuda_devices() {
        let set = DeviceSet::cuda(0..3).expect("an ordinal range is a valid set");
        assert_eq!(
            set.devices(),
            [DeviceId::cuda(0), DeviceId::cuda(1), DeviceId::cuda(2)]
        );
        assert_eq!(set.kind(), Some(DeviceKind::Cuda));
    }

    #[test]
    fn an_empty_ordinal_range_is_rejected_rather_than_defaulted() {
        assert_eq!(DeviceSet::cuda(0..0), Err(DeviceSetError::Empty));
        assert_eq!(DeviceSet::wgpu(0..0), Err(DeviceSetError::Empty));
    }

    /// The default has to be the boring one. A `Fastest` default would move an
    /// unchanged program onto a GPU the day one appears.
    #[test]
    fn the_default_preference_is_the_cpu() {
        assert_eq!(DevicePreference::default(), DevicePreference::Cpu);
    }

    #[test]
    fn test_device_variants() {
        let cpu = DeviceId::cpu();
        assert_eq!(cpu.kind(), DeviceKind::Cpu);
        assert_eq!(cpu.ordinal(), 0);

        #[cfg(feature = "cuda")]
        {
            let cuda = DeviceId::cuda(0);
            assert_eq!(cuda.kind(), DeviceKind::Cuda);
            assert_eq!(cuda.ordinal(), 0);

            let cuda2 = DeviceId::cuda(2);
            assert_eq!(cuda2.ordinal(), 2);
        }

        #[cfg(feature = "wgpu")]
        {
            let wgpu = DeviceId::wgpu(0);
            assert_eq!(wgpu.kind(), DeviceKind::Wgpu);
            assert_eq!(wgpu.ordinal(), 0);
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_tier2_runtime_ordinal() {
        // Tier 2: kind known at compile time, ordinal at runtime
        let field = <Cuda as Device>::init(Cuda::new(3));
        assert_eq!(field.ordinal(), 3);
        let id = Cuda::to_incin(&field).unwrap();
        assert_eq!(id, DeviceId::cuda(3));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_tier3_static_ordinal() {
        use typenum::U2;
        // Tier 3: both kind and ordinal known at compile time
        let field = <CudaN<U2> as Device>::init(());
        let id = CudaN::<U2>::to_incin(&field).unwrap();
        assert_eq!(id, DeviceId::cuda(2));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn test_wgpu_tier2_runtime_ordinal() {
        let field = <Wgpu as Device>::init(Wgpu::new(1));
        assert_eq!(field.ordinal(), 1);
        let id = Wgpu::to_incin(&field).unwrap();
        assert_eq!(id, DeviceId::wgpu(1));
    }

    #[cfg(feature = "metal")]
    #[test]
    fn test_metal_tier2_runtime_ordinal() {
        let field = <Metal as Device>::init(Metal::new(2));
        assert_eq!(field.ordinal(), 2);
        assert_eq!(Metal::to_incin(&field).unwrap(), DeviceId::metal(2));
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn test_wgpu_tier3_static_ordinal() {
        use typenum::U0;
        let field = <WgpuN<U0> as Device>::init(());
        let id = WgpuN::<U0>::to_incin(&field).unwrap();
        assert_eq!(id, DeviceId::wgpu(0));
    }

    #[test]
    fn test_dyn_tier1_fully_runtime() {
        // Tier 1: both kind and ordinal at runtime
        let id = DeviceId::cuda(5);
        let field = <Dyn as Device>::init(id);
        assert_eq!(Dyn::to_incin(&field).unwrap(), DeviceId::cuda(5));
    }

    #[test]
    fn external_device_identity_is_open_and_stable() {
        let id = DeviceId::custom(0x434f_4d50_414e_5901, 7);
        assert_eq!(id.kind().name(), "custom");
        assert_eq!(id.kind().custom_key(), Some(0x434f_4d50_414e_5901));
        assert_eq!(id.ordinal(), 7);
    }
}
