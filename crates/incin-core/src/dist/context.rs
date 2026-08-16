//! Two-rank distributed process identity, rendezvous, and fail-stop lifecycle.
//!
//! A static context proves its logical mesh has exactly two ranks and its
//! process rank is either [`U0`] or [`U1`]. [`Dyn`] accepts the same choices
//! only after checking them at runtime. Hardware existence remains a backend
//! concern: the launcher records the process-local CUDA ordinal, but does not
//! claim that the device exists before the CUDA backend opens it.
//!
//! TCP rendezvous is available only with `std`. The control connection remains
//! open after startup so either process can invalidate the job and both can
//! perform a bounded coordinated shutdown. This is deliberately separate from
//! communicator bootstrap: rendezvous proves process identity and lifecycle,
//! while a transport still proves its own plan and communicator identity.

use alloc::string::String;
#[cfg(feature = "std")]
use alloc::string::ToString;
use alloc::sync::Arc;
#[cfg(feature = "std")]
use alloc::vec::Vec;
use core::fmt;
use core::marker::PhantomData;
use core::sync::atomic::{AtomicU8, Ordering};

#[cfg(feature = "std")]
use typenum::U2;
use typenum::{U0, U1, Unsigned};

#[cfg(feature = "std")]
use crate::dist::mesh::ValidMesh;
use crate::shapes::Dyn;

#[cfg(feature = "std")]
use std::io::{Read, Write};
#[cfg(feature = "std")]
use std::net::{SocketAddr, TcpListener, TcpStream};
#[cfg(feature = "std")]
use std::process::Command;
#[cfg(feature = "std")]
use std::sync::Mutex;
#[cfg(feature = "std")]
use std::time::{Duration, Instant};

/// The only world size accepted by the first network launcher.
pub const TWO_RANK_WORLD: usize = 2;

/// Environment variable containing this process's rank.
pub const RANK_ENV: &str = "INCIN_RANK";
/// Environment variable containing the exact world size.
pub const WORLD_SIZE_ENV: &str = "INCIN_WORLD_SIZE";
/// Environment variable containing rank zero's reachable socket address.
pub const RENDEZVOUS_ADDR_ENV: &str = "INCIN_RENDEZVOUS_ADDR";
/// Environment variable containing the shared, non-secret run identity.
pub const RUN_ID_ENV: &str = "INCIN_RUN_ID";
/// Environment variable containing the process-local CUDA ordinal.
pub const LOCAL_CUDA_DEVICE_ENV: &str = "INCIN_LOCAL_CUDA_DEVICE";
/// Environment variable containing the rendezvous deadline in milliseconds.
pub const RENDEZVOUS_TIMEOUT_MS_ENV: &str = "INCIN_RENDEZVOUS_TIMEOUT_MS";

const MAX_RUN_ID_BYTES: usize = 128;
const STATE_ACTIVE: u8 = 0;
const STATE_SHUTTING_DOWN: u8 = 1;
const STATE_SHUTDOWN: u8 = 2;
const STATE_FAILED: u8 = 3;

/// A type-level rank admitted by an exactly two-rank context.
///
/// This trait is sealed to [`U0`] and [`U1`]. A static `U2` rank therefore
/// fails to compile instead of reaching a runtime branch.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid rank in a two-rank distributed context",
    label = "expected typenum::U0 or typenum::U1"
)]
pub trait StaticTwoRank: sealed::SealedRank + Unsigned + 'static {
    /// The rank encoded by this marker.
    const RANK: usize;
}

impl StaticTwoRank for U0 {
    const RANK: usize = 0;
}

impl StaticTwoRank for U1 {
    const RANK: usize = 1;
}

mod sealed {
    use typenum::{U0, U1};

    pub trait SealedRank {}
    impl SealedRank for U0 {}
    impl SealedRank for U1 {}
}

/// Stable, shared identity for one distributed run.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RunId(String);

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
    const fn decode(value: u8) -> Self {
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
    const fn from_code(code: u16) -> Self {
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
    state: Arc<AtomicU8>,
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

/// Identity agreed during rendezvous.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DistributedIdentity {
    run_id: RunId,
    rank: usize,
    world: usize,
    local_cuda_device: usize,
    peer_cuda_device: usize,
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

/// An agreed process-per-rank context.
///
/// `M` is either a static [`ValidMesh`] with `World = U2` or [`Dyn`]. `R` is
/// either [`U0`]/[`U1`] or [`Dyn`]. The default is runtime selection because
/// environment variables necessarily arrive at runtime.
pub struct DistributedContext<M = Dyn, R = Dyn> {
    identity: DistributedIdentity,
    handle: DistributedContextHandle,
    #[cfg(feature = "std")]
    control: Arc<Mutex<TcpStream>>,
    #[cfg(feature = "std")]
    endpoint: RendezvousEndpoint,
    #[cfg(feature = "std")]
    timeout: Duration,
    marker: PhantomData<(M, R)>,
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

#[cfg(feature = "std")]
impl<M, R> DistributedContext<M, R>
where
    M: ValidMesh<World = U2>,
    R: StaticTwoRank,
{
    /// Perform TCP rendezvous with compile-time world and rank proofs.
    pub fn rendezvous_static(config: StaticRendezvousConfig<R>) -> Result<Self, ContextError> {
        rendezvous(
            config.run_id,
            config.endpoint,
            R::RANK,
            TWO_RANK_WORLD,
            config.local_cuda_device,
            config.timeout,
        )
    }
}

#[cfg(feature = "std")]
impl DistributedContext<Dyn, Dyn> {
    /// Perform TCP rendezvous after runtime validation of every dynamic field.
    pub fn rendezvous_dyn(config: DynRendezvousConfig) -> Result<Self, ContextError> {
        validate_dyn_launch(config.endpoint, config.rank, config.world, config.timeout)?;
        rendezvous(
            config.run_id,
            config.endpoint,
            config.rank,
            config.world,
            config.local_cuda_device,
            config.timeout,
        )
    }

    /// Build and rendezvous a dynamic context from scheduler-friendly
    /// environment variables.
    pub fn from_env() -> Result<Self, ContextError> {
        Self::rendezvous_dyn(DynRendezvousConfig::from_env()?)
    }
}

/// Rank zero listens; rank one connects to the same reachable address.
#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RendezvousEndpoint {
    /// Rank zero's bind address.
    Root { bind: SocketAddr },
    /// Rank zero's address as reachable from rank one.
    Peer { root: SocketAddr },
}

#[cfg(feature = "std")]
impl RendezvousEndpoint {
    /// Rank implied by this role.
    #[must_use]
    pub const fn rank(self) -> usize {
        match self {
            Self::Root { .. } => 0,
            Self::Peer { .. } => 1,
        }
    }

    /// Shared root address.
    #[must_use]
    pub const fn address(self) -> SocketAddr {
        match self {
            Self::Root { bind } => bind,
            Self::Peer { root } => root,
        }
    }
}

/// Static rendezvous configuration. Its constructor fixes the rank in `R`.
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct StaticRendezvousConfig<R> {
    run_id: RunId,
    endpoint: RendezvousEndpoint,
    local_cuda_device: usize,
    timeout: Duration,
    marker: PhantomData<R>,
}

#[cfg(feature = "std")]
impl StaticRendezvousConfig<U0> {
    /// Configure static rank zero.
    #[must_use]
    pub const fn root(
        run_id: RunId,
        bind: SocketAddr,
        local_cuda_device: usize,
        timeout: Duration,
    ) -> Self {
        Self {
            run_id,
            endpoint: RendezvousEndpoint::Root { bind },
            local_cuda_device,
            timeout,
            marker: PhantomData,
        }
    }
}

#[cfg(feature = "std")]
impl StaticRendezvousConfig<U1> {
    /// Configure static rank one.
    #[must_use]
    pub const fn peer(
        run_id: RunId,
        root: SocketAddr,
        local_cuda_device: usize,
        timeout: Duration,
    ) -> Self {
        Self {
            run_id,
            endpoint: RendezvousEndpoint::Peer { root },
            local_cuda_device,
            timeout,
            marker: PhantomData,
        }
    }
}

/// Runtime-selected rendezvous configuration.
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct DynRendezvousConfig {
    run_id: RunId,
    endpoint: RendezvousEndpoint,
    rank: usize,
    world: usize,
    local_cuda_device: usize,
    timeout: Duration,
}

#[cfg(feature = "std")]
impl DynRendezvousConfig {
    /// Store runtime-selected fields. [`DistributedContext::rendezvous_dyn`]
    /// validates them before opening a socket.
    #[must_use]
    pub const fn new(
        run_id: RunId,
        endpoint: RendezvousEndpoint,
        rank: usize,
        world: usize,
        local_cuda_device: usize,
        timeout: Duration,
    ) -> Self {
        Self {
            run_id,
            endpoint,
            rank,
            world,
            local_cuda_device,
            timeout,
        }
    }

    /// Read the six documented launcher variables without opening a socket.
    pub fn from_env() -> Result<Self, ContextError> {
        let run_id = RunId::new(read_env(RUN_ID_ENV)?)?;
        let rank = parse_env::<usize>(RANK_ENV)?;
        let world = parse_env::<usize>(WORLD_SIZE_ENV)?;
        let address = parse_env::<SocketAddr>(RENDEZVOUS_ADDR_ENV)?;
        let local_cuda_device = parse_env::<usize>(LOCAL_CUDA_DEVICE_ENV)?;
        let timeout_ms = parse_env::<u64>(RENDEZVOUS_TIMEOUT_MS_ENV)?;
        let endpoint = match rank {
            0 => RendezvousEndpoint::Root { bind: address },
            _ => RendezvousEndpoint::Peer { root: address },
        };
        Ok(Self::new(
            run_id,
            endpoint,
            rank,
            world,
            local_cuda_device,
            Duration::from_millis(timeout_ms),
        ))
    }
}

/// One rank's explicit launch environment.
#[cfg(feature = "std")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankLaunch {
    run_id: RunId,
    root: SocketAddr,
    rank: usize,
    world: usize,
    local_cuda_device: usize,
    timeout_ms: u64,
}

#[cfg(feature = "std")]
impl RankLaunch {
    /// Rank represented by this launch record.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Process-local CUDA ordinal.
    #[must_use]
    pub const fn local_cuda_device(&self) -> usize {
        self.local_cuda_device
    }

    /// Exact environment passed to a scheduler or child process.
    #[must_use]
    pub fn environment(&self) -> Vec<(&'static str, String)> {
        vec![
            (RUN_ID_ENV, self.run_id.as_str().to_string()),
            (RANK_ENV, self.rank.to_string()),
            (WORLD_SIZE_ENV, self.world.to_string()),
            (RENDEZVOUS_ADDR_ENV, self.root.to_string()),
            (LOCAL_CUDA_DEVICE_ENV, self.local_cuda_device.to_string()),
            (RENDEZVOUS_TIMEOUT_MS_ENV, self.timeout_ms.to_string()),
        ]
    }

    /// Apply this launch record to an ordinary [`Command`].
    pub fn apply(&self, command: &mut Command) {
        command.envs(self.environment());
    }
}

/// A launcher plan for two process-local CUDA devices.
///
/// The devices may both be ordinal zero: ordinals are local to each network
/// host and are not treated as persistent cross-host identities.
#[cfg(feature = "std")]
#[derive(Debug, Clone)]
pub struct TwoRankLaunchPlan<M = Dyn> {
    run_id: RunId,
    root: SocketAddr,
    local_cuda_devices: [usize; TWO_RANK_WORLD],
    timeout: Duration,
    marker: PhantomData<M>,
}

#[cfg(feature = "std")]
impl<M> TwoRankLaunchPlan<M>
where
    M: ValidMesh<World = U2>,
{
    /// Build a launcher whose world cardinality is proved by `M`.
    pub fn new_static(
        run_id: RunId,
        root: SocketAddr,
        local_cuda_devices: [usize; TWO_RANK_WORLD],
        timeout: Duration,
    ) -> Result<Self, ContextError> {
        validate_timeout(timeout)?;
        Ok(Self {
            run_id,
            root,
            local_cuda_devices,
            timeout,
            marker: PhantomData,
        })
    }

    /// Produce a launch record for a statically valid rank.
    #[must_use]
    pub fn rank_static<R: StaticTwoRank>(&self) -> RankLaunch {
        self.rank_unchecked(R::RANK)
    }
}

#[cfg(feature = "std")]
impl TwoRankLaunchPlan<Dyn> {
    /// Build a launcher after checking a runtime-selected world size.
    pub fn new_dyn(
        run_id: RunId,
        root: SocketAddr,
        world: usize,
        local_cuda_devices: Vec<usize>,
        timeout: Duration,
    ) -> Result<Self, ContextError> {
        if world != TWO_RANK_WORLD {
            return Err(ContextError::WorldSize {
                expected: TWO_RANK_WORLD,
                found: world,
            });
        }
        if local_cuda_devices.len() != TWO_RANK_WORLD {
            return Err(ContextError::DeviceCount {
                expected: TWO_RANK_WORLD,
                found: local_cuda_devices.len(),
            });
        }
        validate_timeout(timeout)?;
        Ok(Self {
            run_id,
            root,
            local_cuda_devices: [local_cuda_devices[0], local_cuda_devices[1]],
            timeout,
            marker: PhantomData,
        })
    }

    /// Produce a launch record after checking the runtime rank.
    pub fn rank_dyn(&self, rank: usize) -> Result<RankLaunch, ContextError> {
        if rank >= TWO_RANK_WORLD {
            return Err(ContextError::RankOutOfRange {
                rank,
                world: TWO_RANK_WORLD,
            });
        }
        Ok(self.rank_unchecked(rank))
    }
}

#[cfg(feature = "std")]
impl<M> TwoRankLaunchPlan<M> {
    fn rank_unchecked(&self, rank: usize) -> RankLaunch {
        RankLaunch {
            run_id: self.run_id.clone(),
            root: self.root,
            rank,
            world: TWO_RANK_WORLD,
            local_cuda_device: self.local_cuda_devices[rank],
            timeout_ms: self.timeout.as_millis().min(u128::from(u64::MAX)) as u64,
        }
    }
}

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
fn validate_timeout(timeout: Duration) -> Result<(), ContextError> {
    if timeout.is_zero() || timeout.as_millis() > u128::from(u64::MAX) {
        Err(ContextError::InvalidTimeout)
    } else {
        Ok(())
    }
}

#[cfg(feature = "std")]
fn validate_dyn_launch(
    endpoint: RendezvousEndpoint,
    rank: usize,
    world: usize,
    timeout: Duration,
) -> Result<(), ContextError> {
    if world != TWO_RANK_WORLD {
        return Err(ContextError::WorldSize {
            expected: TWO_RANK_WORLD,
            found: world,
        });
    }
    if rank >= world {
        return Err(ContextError::RankOutOfRange { rank, world });
    }
    if endpoint.rank() != rank {
        return Err(ContextError::RoleRankMismatch {
            role_rank: endpoint.rank(),
            rank,
        });
    }
    validate_timeout(timeout)
}

#[cfg(feature = "std")]
fn read_env(name: &'static str) -> Result<String, ContextError> {
    std::env::var(name).map_err(|_| ContextError::MissingEnvironment { name })
}

#[cfg(feature = "std")]
fn parse_env<T>(name: &'static str) -> Result<T, ContextError>
where
    T: core::str::FromStr,
{
    let value = read_env(name)?;
    value
        .parse()
        .map_err(|_| ContextError::InvalidEnvironment { name, value })
}

#[cfg(feature = "std")]
fn rendezvous<M, R>(
    run_id: RunId,
    endpoint: RendezvousEndpoint,
    rank: usize,
    world: usize,
    local_cuda_device: usize,
    timeout: Duration,
) -> Result<DistributedContext<M, R>, ContextError> {
    validate_dyn_launch(endpoint, rank, world, timeout)?;
    let local = StartupMessage::hello(&run_id, rank, local_cuda_device)?;
    let (stream, remote) = match endpoint {
        RendezvousEndpoint::Root { bind } => {
            let listener =
                TcpListener::bind(bind).map_err(|error| network("bind rendezvous", error))?;
            listener
                .set_nonblocking(true)
                .map_err(|error| network("configure rendezvous listener", error))?;
            let mut stream = accept_until(&listener, timeout)?;
            drop(listener);
            configure_stream(&stream, timeout)?;
            let remote = read_startup(&mut stream)?;
            if let Err(error) = validate_startup(&remote, &run_id, 1) {
                let _ = write_startup(&mut stream, StartupMessage::reject(rejection_code(&error)));
                return Err(error);
            }
            write_startup(
                &mut stream,
                StartupMessage::accepted(&run_id, rank, local_cuda_device)?,
            )?;
            (stream, remote)
        }
        RendezvousEndpoint::Peer { root } => {
            let mut stream = connect_until(root, timeout)?;
            configure_stream(&stream, timeout)?;
            write_startup(&mut stream, local)?;
            let remote = read_startup(&mut stream)?;
            if remote.kind == MessageKind::Reject {
                return Err(ContextError::PeerRejected { code: remote.code });
            }
            validate_startup(&remote, &run_id, 0)?;
            if remote.kind != MessageKind::Accepted {
                return Err(ContextError::Protocol(
                    "rank zero did not accept rendezvous",
                ));
            }
            (stream, remote)
        }
    };
    configure_stream(&stream, timeout)?;
    // Disable Nagle for the tiny shutdown/abort messages. Failure is reported:
    // a context is not considered active unless its control path is usable.
    stream
        .set_nodelay(true)
        .map_err(|error| network("configure rendezvous control socket", error))?;

    Ok(DistributedContext {
        identity: DistributedIdentity {
            run_id,
            rank,
            world,
            local_cuda_device,
            peer_cuda_device: remote.local_cuda_device as usize,
        },
        handle: DistributedContextHandle {
            state: Arc::new(AtomicU8::new(STATE_ACTIVE)),
        },
        control: Arc::new(Mutex::new(stream)),
        endpoint,
        timeout,
        marker: PhantomData,
    })
}

#[cfg(feature = "std")]
fn rejection_code(error: &ContextError) -> u16 {
    match error {
        ContextError::RunIdMismatch => 1,
        ContextError::RemoteRank { .. } => 2,
        ContextError::WorldSize { .. } => 3,
        _ => 255,
    }
}

#[cfg(feature = "std")]
fn network(phase: &'static str, error: std::io::Error) -> ContextError {
    ContextError::Network {
        phase,
        message: error.to_string(),
    }
}

#[cfg(feature = "std")]
fn configure_stream(stream: &TcpStream, timeout: Duration) -> Result<(), ContextError> {
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|error| network("set rendezvous read timeout", error))?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|error| network("set rendezvous write timeout", error))
}

#[cfg(feature = "std")]
fn accept_until(listener: &TcpListener, timeout: Duration) -> Result<TcpStream, ContextError> {
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or(ContextError::InvalidTimeout)?;
    loop {
        match listener.accept() {
            Ok((stream, _)) => return Ok(stream),
            Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Err(ContextError::Network {
                        phase: "accept rank one",
                        message: "rendezvous deadline elapsed".to_string(),
                    });
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(error) => return Err(network("accept rank one", error)),
        }
    }
}

#[cfg(feature = "std")]
fn connect_until(root: SocketAddr, timeout: Duration) -> Result<TcpStream, ContextError> {
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or(ContextError::InvalidTimeout)?;
    loop {
        match TcpStream::connect_timeout(&root, timeout.min(Duration::from_millis(50))) {
            Ok(stream) => return Ok(stream),
            Err(error)
                if matches!(
                    error.kind(),
                    std::io::ErrorKind::ConnectionRefused
                        | std::io::ErrorKind::TimedOut
                        | std::io::ErrorKind::NotConnected
                ) =>
            {
                if Instant::now() >= deadline {
                    return Err(ContextError::Network {
                        phase: "connect to rank zero",
                        message: "rendezvous deadline elapsed".to_string(),
                    });
                }
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(error) => return Err(network("connect to rank zero", error)),
        }
    }
}

#[cfg(feature = "std")]
const PROTOCOL_MAGIC: [u8; 8] = *b"INCINRV1";
#[cfg(feature = "std")]
const STARTUP_BYTES: usize = 160;
#[cfg(feature = "std")]
const CONTROL_BYTES: usize = 16;

#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
enum MessageKind {
    Hello = 1,
    Accepted = 2,
    Reject = 3,
    Shutdown = 4,
    Abort = 5,
}

#[cfg(feature = "std")]
impl MessageKind {
    fn decode(value: u8) -> Result<Self, ContextError> {
        match value {
            1 => Ok(Self::Hello),
            2 => Ok(Self::Accepted),
            3 => Ok(Self::Reject),
            4 => Ok(Self::Shutdown),
            5 => Ok(Self::Abort),
            _ => Err(ContextError::Protocol("unknown rendezvous message kind")),
        }
    }
}

#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy)]
struct StartupMessage {
    kind: MessageKind,
    rank: u8,
    world: u8,
    code: u16,
    local_cuda_device: u64,
    run_id_len: u16,
    run_id: [u8; MAX_RUN_ID_BYTES],
}

#[cfg(feature = "std")]
impl StartupMessage {
    fn hello(run_id: &RunId, rank: usize, local_cuda_device: usize) -> Result<Self, ContextError> {
        Self::new(MessageKind::Hello, run_id, rank, local_cuda_device)
    }

    fn accepted(
        run_id: &RunId,
        rank: usize,
        local_cuda_device: usize,
    ) -> Result<Self, ContextError> {
        Self::new(MessageKind::Accepted, run_id, rank, local_cuda_device)
    }

    fn new(
        kind: MessageKind,
        run_id: &RunId,
        rank: usize,
        local_cuda_device: usize,
    ) -> Result<Self, ContextError> {
        let mut bytes = [0; MAX_RUN_ID_BYTES];
        bytes[..run_id.0.len()].copy_from_slice(run_id.0.as_bytes());
        Ok(Self {
            kind,
            rank: u8::try_from(rank)
                .map_err(|_| ContextError::RankOutOfRange { rank, world: 2 })?,
            world: TWO_RANK_WORLD as u8,
            code: 0,
            local_cuda_device: u64::try_from(local_cuda_device).map_err(|_| {
                ContextError::Protocol("local CUDA ordinal does not fit rendezvous wire")
            })?,
            run_id_len: run_id.0.len() as u16,
            run_id: bytes,
        })
    }

    const fn reject(code: u16) -> Self {
        Self {
            kind: MessageKind::Reject,
            rank: 0,
            world: TWO_RANK_WORLD as u8,
            code,
            local_cuda_device: 0,
            run_id_len: 0,
            run_id: [0; MAX_RUN_ID_BYTES],
        }
    }

    fn encode(self) -> [u8; STARTUP_BYTES] {
        let mut bytes = [0; STARTUP_BYTES];
        bytes[..8].copy_from_slice(&PROTOCOL_MAGIC);
        bytes[8] = self.kind as u8;
        bytes[9] = self.rank;
        bytes[10] = self.world;
        bytes[12..14].copy_from_slice(&self.code.to_be_bytes());
        bytes[16..24].copy_from_slice(&self.local_cuda_device.to_be_bytes());
        bytes[24..26].copy_from_slice(&self.run_id_len.to_be_bytes());
        bytes[32..].copy_from_slice(&self.run_id);
        bytes
    }

    fn decode(bytes: [u8; STARTUP_BYTES]) -> Result<Self, ContextError> {
        if bytes[..8] != PROTOCOL_MAGIC {
            return Err(ContextError::Protocol("rendezvous magic mismatch"));
        }
        let run_id_len = u16::from_be_bytes([bytes[24], bytes[25]]);
        if usize::from(run_id_len) > MAX_RUN_ID_BYTES {
            return Err(ContextError::Protocol(
                "rendezvous run identity is too long",
            ));
        }
        let mut run_id = [0; MAX_RUN_ID_BYTES];
        run_id.copy_from_slice(&bytes[32..]);
        Ok(Self {
            kind: MessageKind::decode(bytes[8])?,
            rank: bytes[9],
            world: bytes[10],
            code: u16::from_be_bytes([bytes[12], bytes[13]]),
            local_cuda_device: u64::from_be_bytes(
                bytes[16..24]
                    .try_into()
                    .map_err(|_| ContextError::Protocol("invalid rendezvous device field"))?,
            ),
            run_id_len,
            run_id,
        })
    }

    fn run_id(&self) -> &[u8] {
        &self.run_id[..usize::from(self.run_id_len)]
    }
}

#[cfg(feature = "std")]
fn validate_startup(
    message: &StartupMessage,
    run_id: &RunId,
    expected_rank: usize,
) -> Result<(), ContextError> {
    if message.world as usize != TWO_RANK_WORLD {
        return Err(ContextError::WorldSize {
            expected: TWO_RANK_WORLD,
            found: message.world as usize,
        });
    }
    if message.rank as usize != expected_rank {
        return Err(ContextError::RemoteRank {
            expected: expected_rank,
            found: message.rank as usize,
        });
    }
    if message.run_id() != run_id.as_str().as_bytes() {
        return Err(ContextError::RunIdMismatch);
    }
    if !matches!(message.kind, MessageKind::Hello | MessageKind::Accepted) {
        return Err(ContextError::Protocol(
            "unexpected message during rendezvous",
        ));
    }
    Ok(())
}

#[cfg(feature = "std")]
fn write_startup(stream: &mut TcpStream, message: StartupMessage) -> Result<(), ContextError> {
    stream
        .write_all(&message.encode())
        .map_err(|error| network("write rendezvous startup", error))
}

#[cfg(feature = "std")]
fn read_startup(stream: &mut TcpStream) -> Result<StartupMessage, ContextError> {
    let mut bytes = [0; STARTUP_BYTES];
    stream
        .read_exact(&mut bytes)
        .map_err(|error| network("read rendezvous startup", error))?;
    StartupMessage::decode(bytes)
}

#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy)]
struct ControlMessage {
    kind: MessageKind,
    code: u16,
}

#[cfg(feature = "std")]
impl ControlMessage {
    const fn shutdown() -> Self {
        Self {
            kind: MessageKind::Shutdown,
            code: 0,
        }
    }

    const fn abort(failure: ContextFailure) -> Self {
        Self {
            kind: MessageKind::Abort,
            code: failure as u16,
        }
    }

    fn encode(self) -> [u8; CONTROL_BYTES] {
        let mut bytes = [0; CONTROL_BYTES];
        bytes[..8].copy_from_slice(&PROTOCOL_MAGIC);
        bytes[8] = self.kind as u8;
        bytes[10..12].copy_from_slice(&self.code.to_be_bytes());
        bytes
    }

    fn decode(bytes: [u8; CONTROL_BYTES]) -> Result<Self, ContextError> {
        if bytes[..8] != PROTOCOL_MAGIC {
            return Err(ContextError::Protocol("control message magic mismatch"));
        }
        Ok(Self {
            kind: MessageKind::decode(bytes[8])?,
            code: u16::from_be_bytes([bytes[10], bytes[11]]),
        })
    }
}

#[cfg(feature = "std")]
fn write_control(stream: &mut TcpStream, message: ControlMessage) -> Result<(), ContextError> {
    stream
        .write_all(&message.encode())
        .map_err(|error| network("write rendezvous control message", error))
}

#[cfg(feature = "std")]
fn read_control(stream: &mut TcpStream) -> Result<ControlMessage, ContextError> {
    let mut bytes = [0; CONTROL_BYTES];
    stream
        .read_exact(&mut bytes)
        .map_err(|error| network("read rendezvous control message", error))?;
    ControlMessage::decode(bytes)
}
