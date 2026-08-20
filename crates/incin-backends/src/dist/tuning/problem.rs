use super::*;

/// Stable cache identity for a collective tuning problem.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CollectiveTuningKey {
    pub(super) kind: CollectiveKind,
    pub(super) dtype: DTypeId,
    pub(super) message_size_bucket: u8,
    pub(super) group: GroupId,
    pub(super) topology: u64,
    pub(super) transport: String,
    pub(super) transport_version: (u32, u32, u32),
    pub(super) determinism: Determinism,
    pub(super) hash: u64,
}

impl CollectiveTuningKey {
    /// Collective semantics.
    #[must_use]
    pub const fn kind(&self) -> CollectiveKind {
        self.kind
    }

    /// Static projection or runtime dtype.
    #[must_use]
    pub const fn dtype(&self) -> DTypeId {
        self.dtype
    }

    /// Ceiling-log2 byte bucket.
    #[must_use]
    pub const fn message_size_bucket(&self) -> u8 {
        self.message_size_bucket
    }

    /// Exact ordered group.
    #[must_use]
    pub const fn group(&self) -> GroupId {
        self.group
    }

    /// Stable topology digest including rank-to-device mapping.
    #[must_use]
    pub const fn topology(&self) -> u64 {
        self.topology
    }

    /// Communication library.
    #[must_use]
    pub fn transport(&self) -> &str {
        &self.transport
    }

    /// Communication library version.
    #[must_use]
    pub const fn transport_version(&self) -> (u32, u32, u32) {
        self.transport_version
    }

    /// Determinism filter.
    #[must_use]
    pub const fn determinism(&self) -> Determinism {
        self.determinism
    }

    /// Stable cross-rank identity.
    #[must_use]
    pub const fn hash(&self) -> u64 {
        self.hash
    }
}

/// Fully validated two-rank collective tuning problem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollectiveTuningProblem {
    pub(super) key: CollectiveTuningKey,
    pub(super) elements: usize,
    pub(super) message_bytes: usize,
    pub(super) workspace_budget_bytes: usize,
}

impl CollectiveTuningProblem {
    /// Build a problem whose operation, dtype, and element count are static.
    pub fn new_static<K, Elements, Operation>(
        group: GroupId,
        topology: &TopologyFingerprint,
        determinism: Determinism,
        workspace_budget_bytes: usize,
    ) -> Result<Self, CollectiveTuningError>
    where
        K: ConstDType + CollectiveDType,
        Elements: Unsigned + NonZero + IsLessOrEqual<U4294967295, Output = B1>,
        Operation: StaticCollectiveTuning<K, Elements>,
    {
        Self::new_dyn(
            Operation::KIND,
            K::DESCRIPTOR.builtin_id().unwrap_or(DTypeId::F32),
            Elements::USIZE,
            group,
            topology,
            determinism,
            workspace_budget_bytes,
        )
    }

    /// Build the matching runtime-selected (`Dyn`) problem.
    #[allow(clippy::too_many_arguments)]
    pub fn new_dyn(
        kind: CollectiveKind,
        dtype: DTypeId,
        elements: usize,
        group: GroupId,
        topology: &TopologyFingerprint,
        determinism: Determinism,
        workspace_budget_bytes: usize,
    ) -> Result<Self, CollectiveTuningError> {
        if group.ranks() != 2 {
            return Err(CollectiveTuningError::GroupSize {
                expected: 2,
                found: group.ranks(),
            });
        }
        if topology.devices().len() != group.ranks() {
            return Err(CollectiveTuningError::TopologyWorld {
                expected: group.ranks(),
                found: topology.devices().len(),
            });
        }
        if elements == 0 {
            return Err(CollectiveTuningError::ZeroElements);
        }
        if elements > u32::MAX as usize {
            return Err(CollectiveTuningError::ElementLimit {
                maximum: u32::MAX as usize,
                found: elements,
            });
        }
        validate_kind(kind, dtype, elements, group.ranks())?;
        let message_bytes = dtype
            .size_bytes(elements, OperationKind::Storage)
            .map_err(|_| CollectiveTuningError::MessageSize)?;
        let transport = topology.transport();
        let message_size_bucket = log2_bucket(message_bytes);
        let transport_version = transport.version();
        let mut key = CollectiveTuningKey {
            kind,
            dtype,
            message_size_bucket,
            group,
            topology: topology.digest(),
            transport: transport.library().into(),
            transport_version,
            determinism,
            hash: 0,
        };
        key.hash = key_hash(&key);
        Ok(Self {
            key,
            elements,
            message_bytes,
            workspace_budget_bytes,
        })
    }

    /// Stable problem/cache identity.
    #[must_use]
    pub const fn key(&self) -> &CollectiveTuningKey {
        &self.key
    }

    /// Exact logical element count used for legality checks.
    #[must_use]
    pub const fn elements(&self) -> usize {
        self.elements
    }

    /// Exact checked message byte count.
    #[must_use]
    pub const fn message_bytes(&self) -> usize {
        self.message_bytes
    }

    /// Maximum transient bytes allowed for a candidate.
    #[must_use]
    pub const fn workspace_budget_bytes(&self) -> usize {
        self.workspace_budget_bytes
    }
}

/// One transport candidate whose representation is stable across ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CollectiveTuningCandidate {
    pub(super) algorithm: CollectiveAlgorithm,
    pub(super) protocol: CollectiveProtocol,
    pub(super) channels: u8,
    pub(super) chunk_bytes: usize,
    pub(super) stream_priority: i8,
    pub(super) overlap_chunks: u8,
    pub(super) workspace_bytes: usize,
    pub(super) deterministic: bool,
    pub(super) hash: u64,
}

impl CollectiveTuningCandidate {
    /// Construct a candidate with statically selected algorithm, protocol,
    /// channel count, and power-of-two chunk size.
    pub fn new_static<Algorithm, Protocol, Channels, ChunkBytes>(
        stream_priority: i8,
        overlap_chunks: u8,
        workspace_bytes: usize,
        deterministic: bool,
    ) -> Result<Self, CollectiveTuningError>
    where
        Algorithm: StaticCollectiveAlgorithm,
        Protocol: StaticCollectiveProtocol,
        Channels: Unsigned + NonZero + IsLessOrEqual<U32, Output = B1>,
        ChunkBytes: Unsigned + NonZero + PowerOfTwo,
    {
        Self::new_dyn(
            Algorithm::ALGORITHM,
            Protocol::PROTOCOL,
            Channels::USIZE,
            ChunkBytes::USIZE,
            stream_priority,
            overlap_chunks,
            workspace_bytes,
            deterministic,
        )
    }

    /// Construct the matching runtime-selected candidate.
    #[allow(clippy::too_many_arguments)]
    pub fn new_dyn(
        algorithm: CollectiveAlgorithm,
        protocol: CollectiveProtocol,
        channels: usize,
        chunk_bytes: usize,
        stream_priority: i8,
        overlap_chunks: u8,
        workspace_bytes: usize,
        deterministic: bool,
    ) -> Result<Self, CollectiveTuningError> {
        if !(1..=32).contains(&channels) {
            return Err(CollectiveTuningError::ChannelCount {
                minimum: 1,
                maximum: 32,
                found: channels,
            });
        }
        if chunk_bytes == 0 || !chunk_bytes.is_power_of_two() {
            return Err(CollectiveTuningError::ChunkBytes { found: chunk_bytes });
        }
        if !(-10..=10).contains(&stream_priority) {
            return Err(CollectiveTuningError::StreamPriority {
                minimum: -10,
                maximum: 10,
                found: stream_priority,
            });
        }
        let channels = channels as u8;
        let mut candidate = Self {
            algorithm,
            protocol,
            channels,
            chunk_bytes,
            stream_priority,
            overlap_chunks,
            workspace_bytes,
            deterministic,
            hash: 0,
        };
        candidate.hash = candidate_hash(candidate);
        Ok(candidate)
    }

    /// Algorithm family.
    #[must_use]
    pub const fn algorithm(self) -> CollectiveAlgorithm {
        self.algorithm
    }

    /// Protocol family.
    #[must_use]
    pub const fn protocol(self) -> CollectiveProtocol {
        self.protocol
    }

    /// Communication channels.
    #[must_use]
    pub const fn channels(self) -> u8 {
        self.channels
    }

    /// Chunk bytes.
    #[must_use]
    pub const fn chunk_bytes(self) -> usize {
        self.chunk_bytes
    }

    /// Requested stream priority.
    #[must_use]
    pub const fn stream_priority(self) -> i8 {
        self.stream_priority
    }

    /// Chunks allowed in the overlap window.
    #[must_use]
    pub const fn overlap_chunks(self) -> u8 {
        self.overlap_chunks
    }

    /// Candidate transient workspace.
    #[must_use]
    pub const fn workspace_bytes(self) -> usize {
        self.workspace_bytes
    }

    /// Whether this candidate may satisfy required determinism.
    #[must_use]
    pub const fn deterministic(self) -> bool {
        self.deterministic
    }

    /// Stable cross-rank identity.
    #[must_use]
    pub const fn hash(self) -> u64 {
        self.hash
    }

    #[cfg(feature = "autotune")]
    /// Converts this transport descriptor into the policy-neutral candidate
    /// metadata consumed by the general tuning service.
    pub fn service_candidate(
        self,
    ) -> Result<crate::tuning::service::TuningCandidate, crate::tuning::service::TuningServiceError>
    {
        crate::tuning::service::TuningCandidate::new(
            self.hash,
            &format!(
                "algorithm={:?},protocol={:?},channels={},chunk_bytes={},stream_priority={},overlap_chunks={},workspace_bytes={},deterministic={}",
                self.algorithm,
                self.protocol,
                self.channels,
                self.chunk_bytes,
                self.stream_priority,
                self.overlap_chunks,
                self.workspace_bytes,
                self.deterministic
            ),
            self.deterministic,
            self.workspace_bytes,
        )
    }
}

/// Hard bound on tuning search and measurement work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CollectiveTuningBudget {
    pub(super) max_candidates: usize,
    pub(super) samples_per_candidate: usize,
}

impl CollectiveTuningBudget {
    /// Build a statically nonzero bounded budget.
    #[must_use]
    pub fn new_static<Candidates, Samples>() -> Self
    where
        Candidates: Unsigned + NonZero + IsLessOrEqual<U32, Output = B1>,
        Samples: Unsigned + NonZero + IsLessOrEqual<U32, Output = B1>,
    {
        Self {
            max_candidates: Candidates::USIZE,
            samples_per_candidate: Samples::USIZE,
        }
    }

    /// Build the runtime-selected budget.
    pub const fn new_dyn(
        max_candidates: usize,
        samples_per_candidate: usize,
    ) -> Result<Self, CollectiveTuningError> {
        if max_candidates == 0 {
            return Err(CollectiveTuningError::ZeroCandidateBudget);
        }
        if samples_per_candidate == 0 {
            return Err(CollectiveTuningError::ZeroSampleBudget);
        }
        if max_candidates > 32 || samples_per_candidate > 32 {
            return Err(CollectiveTuningError::BudgetLimit {
                maximum: 32,
                candidates: max_candidates,
                samples: samples_per_candidate,
            });
        }
        Ok(Self {
            max_candidates,
            samples_per_candidate,
        })
    }

    /// Maximum candidates measured.
    #[must_use]
    pub const fn max_candidates(self) -> usize {
        self.max_candidates
    }

    /// Required synchronized samples from each rank and candidate.
    #[must_use]
    pub const fn samples_per_candidate(self) -> usize {
        self.samples_per_candidate
    }
}
