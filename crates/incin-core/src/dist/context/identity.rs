//! Stable run identity and the agreed per-rank identity produced by
//! rendezvous.

use alloc::string::String;

use super::MAX_RUN_ID_BYTES;
use super::error::ContextError;

/// Stable, shared identity for one distributed run.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RunId(pub(super) String);

impl RunId {
    /// Validate a user- or scheduler-provided run identity.
    pub fn new(value: impl Into<String>) -> Result<Self, ContextError> {
        let value = value.into();
        if value.is_empty() {
            return Err(ContextError::EmptyRunId);
        }
        if value.len() > MAX_RUN_ID_BYTES {
            return Err(ContextError::RunIdTooLong {
                maximum: MAX_RUN_ID_BYTES,
                found: value.len(),
            });
        }
        Ok(Self(value))
    }

    /// The exact identity exchanged by both ranks.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Stable digest suitable for logs and cache identity.
    #[must_use]
    pub fn digest(&self) -> u64 {
        let mut digest = 0xcbf2_9ce4_8422_2325_u64;
        for byte in b"incin.rendezvous.v1"
            .iter()
            .copied()
            .chain(self.0.as_bytes().iter().copied())
        {
            digest ^= u64::from(byte);
            digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
        }
        digest
    }
}

/// Identity agreed during rendezvous.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DistributedIdentity {
    pub(super) run_id: RunId,
    pub(super) rank: usize,
    pub(super) world: usize,
    pub(super) local_cuda_device: usize,
    pub(super) peer_cuda_device: usize,
}

impl DistributedIdentity {
    /// Shared run identity.
    #[must_use]
    pub const fn run_id(&self) -> &RunId {
        &self.run_id
    }

    /// This process's rank.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Agreed world size. It is exactly two for every constructible context.
    #[must_use]
    pub const fn world_size(&self) -> usize {
        self.world
    }

    /// CUDA ordinal local to this process.
    #[must_use]
    pub const fn local_cuda_device(&self) -> usize {
        self.local_cuda_device
    }

    /// CUDA ordinal reported by the peer in its own process namespace.
    #[must_use]
    pub const fn peer_cuda_device(&self) -> usize {
        self.peer_cuda_device
    }
}
