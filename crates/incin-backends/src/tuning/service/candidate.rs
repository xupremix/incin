use super::error::TuningServiceError;
use alloc::string::{String, ToString};

const MAX_CANDIDATE_ENCODING_BYTES: usize = 4096;

/// Candidate metadata the general service can enforce without understanding
/// the backend-specific payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TuningCandidate {
    pub(super) hash: u64,
    pub(super) encoding: String,
    pub(super) deterministic: bool,
    pub(super) workspace_bytes: usize,
}

impl TuningCandidate {
    /// Constructs a candidate with a stable backend-owned hash and encoding.
    pub fn new(
        hash: u64,
        encoding: &str,
        deterministic: bool,
        workspace_bytes: usize,
    ) -> core::result::Result<Self, TuningServiceError> {
        if encoding.is_empty()
            || encoding.len() > MAX_CANDIDATE_ENCODING_BYTES
            || encoding.trim() != encoding
            || encoding.chars().any(char::is_control)
        {
            return Err(TuningServiceError::InvalidCandidateEncoding);
        }
        Ok(Self {
            hash,
            encoding: encoding.to_string(),
            deterministic,
            workspace_bytes,
        })
    }

    /// Stable backend-owned candidate hash.
    #[must_use]
    pub const fn hash(&self) -> u64 {
        self.hash
    }

    /// Persistent encoding which the backend must parse and revalidate.
    #[must_use]
    pub fn encoding(&self) -> &str {
        &self.encoding
    }

    /// Whether this candidate satisfies required determinism.
    #[must_use]
    pub const fn deterministic(&self) -> bool {
        self.deterministic
    }

    /// Transient workspace needed while running the candidate.
    #[must_use]
    pub const fn workspace_bytes(&self) -> usize {
        self.workspace_bytes
    }
}

/// Stable digest of a canonical legal candidate set.
#[must_use]
pub fn legal_candidates_digest(candidates: &[TuningCandidate]) -> u64 {
    let mut ordered = candidates.to_vec();
    ordered.sort_by(|left, right| (left.hash, &left.encoding).cmp(&(right.hash, &right.encoding)));
    let mut digest = Digest::new().field(b"incin.tuning.legal-candidates.v1");
    for candidate in ordered {
        digest = digest
            .number(candidate.hash)
            .text(&candidate.encoding)
            .number(u64::from(candidate.deterministic))
            .number(candidate.workspace_bytes as u64);
    }
    digest.finish()
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

    const fn finish(self) -> u64 {
        self.0
    }
}
