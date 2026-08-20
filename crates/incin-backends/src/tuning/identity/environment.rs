//! Device and compiler identity used by a local-kernel tuning key.

use core::{
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
};

use incin_core::shapes::Dyn;

use super::backend::StaticBackend;
use super::compiler::CompilerFingerprint;
use super::device::DeviceFingerprint;
use super::error::IdentityError;
use super::primitives::{Digest, IDENTITY_SCHEMA};

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
