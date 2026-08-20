//! Stable identity of the compiler which produced a tuned kernel.

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

/// Stable identity of the compiler which produced a tuned kernel.
pub struct CompilerFingerprint<D = Dyn> {
    pub(super) backend: BackendIdentity,
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
