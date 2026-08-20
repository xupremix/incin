//! The agreed per-rank context type and its fail-stop lifecycle surface.

use core::fmt;
use core::marker::PhantomData;
use core::sync::atomic::Ordering;

#[cfg(feature = "std")]
use alloc::sync::Arc;
#[cfg(feature = "std")]
use std::net::TcpStream;
#[cfg(feature = "std")]
use std::sync::Mutex;
#[cfg(feature = "std")]
use std::time::Duration;

use crate::shapes::Dyn;

use super::error::ContextError;
use super::identity::DistributedIdentity;
#[cfg(feature = "std")]
use super::rendezvous::RendezvousEndpoint;
#[cfg(feature = "std")]
use super::state::ContextFailure;
use super::state::{DistributedContextHandle, DistributedContextState};
#[cfg(feature = "std")]
use super::wire::{ControlMessage, MessageKind, read_control, write_control};
use super::{STATE_ACTIVE, STATE_SHUTDOWN, STATE_SHUTTING_DOWN};

/// An agreed process-per-rank context.
///
/// `M` is either a static [`ValidMesh`](crate::dist::mesh::ValidMesh) with
/// `World = U2` or [`Dyn`]. `R` is either [`U0`](typenum::U0)/[`U1`](typenum::U1)
/// or [`Dyn`]. The default is runtime selection because environment variables
/// necessarily arrive at runtime.
pub struct DistributedContext<M = Dyn, R = Dyn> {
    pub(super) identity: DistributedIdentity,
    pub(super) handle: DistributedContextHandle,
    #[cfg(feature = "std")]
    pub(super) control: Arc<Mutex<TcpStream>>,
    #[cfg(feature = "std")]
    pub(super) endpoint: RendezvousEndpoint,
    #[cfg(feature = "std")]
    pub(super) timeout: Duration,
    pub(super) marker: PhantomData<(M, R)>,
}

impl<M, R> fmt::Debug for DistributedContext<M, R> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DistributedContext")
            .field("identity", &self.identity)
            .field("state", &self.state())
            .finish_non_exhaustive()
    }
}

impl<M, R> Clone for DistributedContext<M, R> {
    fn clone(&self) -> Self {
        Self {
            identity: self.identity.clone(),
            handle: self.handle.clone(),
            #[cfg(feature = "std")]
            control: self.control.clone(),
            #[cfg(feature = "std")]
            endpoint: self.endpoint,
            #[cfg(feature = "std")]
            timeout: self.timeout,
            marker: PhantomData,
        }
    }
}

impl<M, R> DistributedContext<M, R> {
    /// Agreed rank, world, run, and process-local device identity.
    #[must_use]
    pub const fn identity(&self) -> &DistributedIdentity {
        &self.identity
    }

    /// This process's rank.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.identity.rank
    }

    /// Exact world size.
    #[must_use]
    pub const fn world_size(&self) -> usize {
        self.identity.world
    }

    /// CUDA ordinal local to this process.
    #[must_use]
    pub const fn local_cuda_device(&self) -> usize {
        self.identity.local_cuda_device
    }

    /// Current fail-stop lifecycle state.
    #[must_use]
    pub fn state(&self) -> DistributedContextState {
        self.handle.state()
    }

    /// Type-erased lifecycle handle suitable for a backend communicator.
    #[must_use]
    pub fn handle(&self) -> DistributedContextHandle {
        self.handle.clone()
    }

    /// Reject work after shutdown or failure.
    pub fn ensure_active(&self) -> Result<(), ContextError> {
        self.handle.ensure_active()
    }

    /// Erase static mesh and rank proofs after they have been established.
    #[must_use]
    pub fn into_dyn(self) -> DistributedContext<Dyn, Dyn> {
        DistributedContext {
            identity: self.identity,
            handle: self.handle,
            #[cfg(feature = "std")]
            control: self.control,
            #[cfg(feature = "std")]
            endpoint: self.endpoint,
            #[cfg(feature = "std")]
            timeout: self.timeout,
            marker: PhantomData,
        }
    }

    /// Rendezvous address and role used by this process.
    #[cfg(feature = "std")]
    #[must_use]
    pub const fn endpoint(&self) -> RendezvousEndpoint {
        self.endpoint
    }

    /// Configured network deadline.
    #[cfg(feature = "std")]
    #[must_use]
    pub const fn timeout(&self) -> Duration {
        self.timeout
    }

    /// Invalidate this context and best-effort notify the peer.
    #[cfg(feature = "std")]
    pub fn abort(&self, failure: ContextFailure) -> Result<(), ContextError> {
        self.handle.invalidate();
        let mut stream = self.lock_control()?;
        write_control(&mut stream, ControlMessage::abort(failure))
            .map_err(|error| self.network_failure(error))
    }

    /// Wait for an abort or shutdown request from the peer.
    ///
    /// A shutdown request is acknowledged before returning. An abort marks the
    /// local context failed and returns [`ContextError::PeerAborted`].
    #[cfg(feature = "std")]
    pub fn wait_for_peer(&self) -> Result<DistributedContextState, ContextError> {
        let mut stream = self.lock_control()?;
        match read_control(&mut stream).map_err(|error| self.network_failure(error))? {
            message if message.kind == MessageKind::Shutdown => {
                write_control(&mut stream, ControlMessage::shutdown())
                    .map_err(|error| self.network_failure(error))?;
                self.handle.state.store(STATE_SHUTDOWN, Ordering::Release);
                Ok(DistributedContextState::Shutdown)
            }
            message if message.kind == MessageKind::Abort => {
                self.handle.invalidate();
                Err(ContextError::PeerAborted {
                    failure: ContextFailure::from_code(message.code),
                })
            }
            _ => {
                self.handle.invalidate();
                Err(ContextError::UnexpectedControlMessage)
            }
        }
    }

    /// Exchange shutdown messages and make further work impossible.
    ///
    /// Both ranks may call this concurrently: TCP is full-duplex, so each
    /// writes before reading the peer's acknowledgement. The socket deadline
    /// bounds a missing or crashed peer.
    #[cfg(feature = "std")]
    pub fn shutdown(&self) -> Result<(), ContextError> {
        match self.handle.state.compare_exchange(
            STATE_ACTIVE,
            STATE_SHUTTING_DOWN,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => {}
            Err(STATE_SHUTDOWN) => return Ok(()),
            Err(value) => {
                return Err(ContextError::ContextNotActive {
                    state: DistributedContextState::decode(value),
                });
            }
        }

        let result = (|| {
            let mut stream = self.lock_control()?;
            write_control(&mut stream, ControlMessage::shutdown())?;
            let response = read_control(&mut stream)?;
            match response.kind {
                MessageKind::Shutdown => Ok(()),
                MessageKind::Abort => Err(ContextError::PeerAborted {
                    failure: ContextFailure::from_code(response.code),
                }),
                _ => Err(ContextError::UnexpectedControlMessage),
            }
        })();

        match result {
            Ok(()) => {
                self.handle.state.store(STATE_SHUTDOWN, Ordering::Release);
                Ok(())
            }
            Err(error) => {
                self.handle.invalidate();
                Err(error)
            }
        }
    }

    #[cfg(feature = "std")]
    fn lock_control(&self) -> Result<std::sync::MutexGuard<'_, TcpStream>, ContextError> {
        self.control
            .lock()
            .map_err(|_| ContextError::ControlLockPoisoned)
    }

    #[cfg(feature = "std")]
    fn network_failure(&self, error: ContextError) -> ContextError {
        self.handle.invalidate();
        error
    }
}
