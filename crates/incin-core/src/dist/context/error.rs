//! Structured validation, rendezvous, and lifecycle failures.

use alloc::string::String;
#[cfg(feature = "std")]
use alloc::string::ToString;

use super::state::{ContextFailure, DistributedContextState};

/// Structured validation, rendezvous, and lifecycle failures.
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ContextError {
    /// Run identities must not be empty.
    #[error("distributed run identity must not be empty")]
    EmptyRunId,
    /// Run identities are bounded so the startup wire message is fixed-size.
    #[error("distributed run identity is {found} bytes; maximum is {maximum}")]
    RunIdTooLong {
        /// Maximum bytes.
        maximum: usize,
        /// Supplied bytes.
        found: usize,
    },
    /// Only two ranks are admitted by this launcher.
    #[error("distributed world size must be {expected}, found {found}")]
    WorldSize {
        /// Required size.
        expected: usize,
        /// Supplied size.
        found: usize,
    },
    /// Dynamic rank must be in `0..world`.
    #[error("rank {rank} is outside world size {world}")]
    RankOutOfRange {
        /// Supplied rank.
        rank: usize,
        /// Supplied world.
        world: usize,
    },
    /// Root/peer role must agree with the runtime rank.
    #[error("rendezvous role implies rank {role_rank}, but configuration says rank {rank}")]
    RoleRankMismatch {
        /// Rank encoded by the endpoint.
        role_rank: usize,
        /// Runtime rank.
        rank: usize,
    },
    /// Runtime launch device count must equal the world size.
    #[error("launcher requires {expected} local-device entries, found {found}")]
    DeviceCount {
        /// Required entries.
        expected: usize,
        /// Supplied entries.
        found: usize,
    },
    /// Zero cannot bound a rendezvous.
    #[error("rendezvous timeout must be nonzero and fit in u64 milliseconds")]
    InvalidTimeout,
    /// A required launcher variable was absent.
    #[error("required launcher environment variable {name} is not set")]
    MissingEnvironment {
        /// Variable name.
        name: &'static str,
    },
    /// A launcher variable could not be parsed.
    #[error("launcher environment variable {name} has invalid value `{value}`")]
    InvalidEnvironment {
        /// Variable name.
        name: &'static str,
        /// Original value.
        value: String,
    },
    /// TCP setup or control I/O failed.
    #[error("{phase} failed: {message}")]
    Network {
        /// Operation in progress.
        phase: &'static str,
        /// Platform error.
        message: String,
    },
    /// The peer sent malformed or incompatible protocol bytes.
    #[error("rendezvous protocol error: {0}")]
    Protocol(&'static str),
    /// The peer's run identity differs.
    #[error("peer joined a different distributed run")]
    RunIdMismatch,
    /// The peer's rank differs from the expected complement.
    #[error("expected remote rank {expected}, found {found}")]
    RemoteRank {
        /// Expected peer.
        expected: usize,
        /// Received peer.
        found: usize,
    },
    /// The root rejected a peer's startup record.
    #[error("rank zero rejected rendezvous with code {code}")]
    PeerRejected {
        /// Stable protocol rejection code.
        code: u16,
    },
    /// Peer explicitly invalidated the run.
    #[error("peer invalidated the distributed context: {failure:?}")]
    PeerAborted {
        /// Reported failure class.
        failure: ContextFailure,
    },
    /// Work was attempted outside the active state.
    #[error("distributed context is {state:?}, not active")]
    ContextNotActive {
        /// Current state.
        state: DistributedContextState,
    },
    /// Another thread panicked while holding the control connection.
    #[error("distributed control connection lock is poisoned")]
    ControlLockPoisoned,
    /// A startup message arrived during the control phase or vice versa.
    #[error("peer sent an unexpected distributed control message")]
    UnexpectedControlMessage,
}

#[cfg(feature = "std")]
pub(super) fn network(phase: &'static str, error: std::io::Error) -> ContextError {
    ContextError::Network {
        phase,
        message: error.to_string(),
    }
}
