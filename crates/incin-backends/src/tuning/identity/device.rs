//! Stable physical-device identity.

use alloc::string::String;
use core::{
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
};

use incin_core::{shapes::Dyn, tensor::device::DeviceKind};

use super::backend::{BackendIdentity, StaticBackend};
use super::error::{IdentityError, checked_field};
use super::primitives::{Digest, IDENTITY_SCHEMA, SoftwareVersion};

/// A stable physical-device identity.
///
/// `D` carries the backend family at compile time for static device markers.
/// `DeviceFingerprint<Dyn>` stores and checks that family at runtime. Device
/// ordinal is intentionally neither accepted nor stored.
pub struct DeviceFingerprint<D = Dyn> {
    pub(super) backend: BackendIdentity,
    pub(super) persistent_id: String,
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

    pub(super) fn physical_key(&self) -> (BackendIdentity, &str) {
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
