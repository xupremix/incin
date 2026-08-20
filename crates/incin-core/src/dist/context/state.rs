//! Runtime lifecycle state and the fail-stop handle shared across clones.

use alloc::sync::Arc;
use core::sync::atomic::{AtomicU8, Ordering};

use super::error::ContextError;
use super::{STATE_ACTIVE, STATE_FAILED, STATE_SHUTDOWN, STATE_SHUTTING_DOWN};

/// Runtime state shared by all clones and transport handles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DistributedContextState {
    /// Rendezvous succeeded and work may be launched.
    Active,
    /// A coordinated shutdown is in progress.
    ShuttingDown,
    /// Both ranks acknowledged shutdown.
    Shutdown,
    /// A local or remote failure invalidated the context.
    Failed,
}

impl DistributedContextState {
    pub(super) const fn decode(value: u8) -> Self {
        match value {
            STATE_ACTIVE => Self::Active,
            STATE_SHUTTING_DOWN => Self::ShuttingDown,
            STATE_SHUTDOWN => Self::Shutdown,
            _ => Self::Failed,
        }
    }
}

/// Why a live context was invalidated.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u16)]
pub enum ContextFailure {
    /// A collective or point-to-point launch failed.
    Transport = 1,
    /// A rank rejected a plan or other distributed identity.
    Agreement = 2,
    /// User code returned an error or panicked at the launcher boundary.
    User = 3,
    /// The peer stopped responding before the deadline.
    Timeout = 4,
    /// A protocol message was malformed or unexpected.
    Protocol = 5,
}

impl ContextFailure {
    #[cfg(feature = "std")]
    pub(super) const fn from_code(code: u16) -> Self {
        match code {
            1 => Self::Transport,
            2 => Self::Agreement,
            3 => Self::User,
            4 => Self::Timeout,
            _ => Self::Protocol,
        }
    }
}

/// A cloneable, type-erased fail-stop handle for backend transports.
///
/// Invalidating this handle invalidates every typed view of the same context.
#[derive(Debug, Clone)]
pub struct DistributedContextHandle {
    pub(super) state: Arc<AtomicU8>,
}

impl DistributedContextHandle {
    /// Current lifecycle state.
    #[must_use]
    pub fn state(&self) -> DistributedContextState {
        DistributedContextState::decode(self.state.load(Ordering::Acquire))
    }

    /// Reject work unless the rendezvous remains active.
    pub fn ensure_active(&self) -> Result<(), ContextError> {
        let state = self.state();
        if state == DistributedContextState::Active {
            Ok(())
        } else {
            Err(ContextError::ContextNotActive { state })
        }
    }

    /// Fail-stop invalidation used by a communicator after an error.
    pub fn invalidate(&self) {
        self.state.store(STATE_FAILED, Ordering::Release);
    }
}
