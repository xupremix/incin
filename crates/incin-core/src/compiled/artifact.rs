//! Versioned compiled artifacts: wraps a `CompiledPlan` with a version header
//! and an Adler-32 checksum for compatibility detection and corruption detection.

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::compiled::plan::CompiledPlan;
use crate::prelude::{Error, Result};

/// Current artifact format version.
pub const ARTIFACT_FORMAT_VERSION: u32 = 1;

/// Magic bytes written at the start of every serialized artifact.
pub const ARTIFACT_MAGIC: [u8; 8] = *b"INCIN\x00\x01\x00";

/// Semantic version of the framework that produced an artifact.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ArtifactVersion {
    /// Framework major version.
    pub major: u32,
    /// Framework minor version.
    pub minor: u32,
    /// Framework patch version.
    pub patch: u32,
    /// Artifact format version — must match `ARTIFACT_FORMAT_VERSION`.
    pub format: u32,
}

impl ArtifactVersion {
    /// Creates a new version descriptor.
    #[must_use]
    pub const fn new(major: u32, minor: u32, patch: u32) -> Self {
        Self {
            major,
            minor,
            patch,
            format: ARTIFACT_FORMAT_VERSION,
        }
    }

    /// Returns `true` if this version is compatible with `other`.
    ///
    /// Compatibility requires matching format and equal major versions.
    #[must_use]
    pub fn is_compatible_with(&self, other: &Self) -> bool {
        self.format == other.format && self.major == other.major
    }
}

/// Header written before the artifact payload.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ArtifactHeader {
    /// Version information.
    pub version: ArtifactVersion,
    /// Adler-32 checksum of the serialized plan bytes.
    pub checksum: u32,
    /// Human-readable label for the artifact.
    pub label: String,
}

/// Computes an Adler-32 checksum over a byte slice.
fn adler32(data: &[u8]) -> u32 {
    const MOD: u32 = 65521;
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &byte in data {
        a = (a + u32::from(byte)) % MOD;
        b = (b + a) % MOD;
    }
    (b << 16) | a
}

/// A versioned, integrity-protected compiled artifact.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CompiledArtifact {
    /// Version and checksum header.
    pub header: ArtifactHeader,
    /// The compiled execution plan.
    pub plan: CompiledPlan,
}

impl CompiledArtifact {
    /// Creates a new artifact from a plan, labeling it and computing a checksum.
    pub fn new(plan: CompiledPlan, version: ArtifactVersion, label: String) -> Result<Self> {
        let plan_bytes = Self::serialize_plan(&plan)?;
        let checksum = adler32(&plan_bytes);
        Ok(Self {
            header: ArtifactHeader {
                version,
                checksum,
                label,
            },
            plan,
        })
    }

    /// Serializes the inner plan to JSON bytes.
    fn serialize_plan(plan: &CompiledPlan) -> Result<Vec<u8>> {
        serde_json::to_vec(plan).map_err(|e| Error::Msg(alloc::format!("serialize: {e}")))
    }

    /// Serializes the entire artifact (header + plan) to JSON bytes.
    pub fn serialize(&self) -> Result<Vec<u8>> {
        serde_json::to_vec(self).map_err(|e| Error::Msg(alloc::format!("serialize: {e}")))
    }

    /// Deserializes an artifact from JSON bytes.
    pub fn deserialize(bytes: &[u8]) -> Result<Self> {
        serde_json::from_slice(bytes).map_err(|e| Error::Msg(alloc::format!("deserialize: {e}")))
    }

    /// Verifies the artifact's integrity by re-computing and comparing the checksum.
    pub fn verify_integrity(&self) -> Result<()> {
        let plan_bytes = Self::serialize_plan(&self.plan)?;
        let computed = adler32(&plan_bytes);
        if computed != self.header.checksum {
            return Err(Error::Msg(alloc::format!(
                "artifact integrity check failed: stored checksum {:#010x}, computed {:#010x}",
                self.header.checksum, computed
            )));
        }
        Ok(())
    }

    /// Checks whether this artifact is compatible with the given required version.
    pub fn check_compatibility(&self, required: &ArtifactVersion) -> Result<()> {
        if !self.header.version.is_compatible_with(required) {
            return Err(Error::Msg(alloc::format!(
                "artifact version {:?} is incompatible with required {:?}",
                self.header.version, required
            )));
        }
        Ok(())
    }

    /// Produces a fresh artifact from `bytes`, verifying both integrity and compatibility.
    pub fn load(bytes: &[u8], required_version: &ArtifactVersion) -> Result<Self> {
        let artifact = Self::deserialize(bytes)?;
        artifact.verify_integrity()?;
        artifact.check_compatibility(required_version)?;
        Ok(artifact)
    }
}
