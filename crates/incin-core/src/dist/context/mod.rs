//! Two-rank distributed process identity, rendezvous, and fail-stop lifecycle.
//!
//! A static context proves its logical mesh has exactly two ranks and its
//! process rank is either [`U0`](typenum::U0) or [`U1`](typenum::U1).
//! [`Dyn`](crate::shapes::Dyn) accepts the same choices only after checking
//! them at runtime. Hardware existence remains a backend concern: the
//! launcher records the process-local CUDA ordinal, but does not claim that
//! the device exists before the CUDA backend opens it.
//!
//! TCP rendezvous is available only with `std`. The control connection remains
//! open after startup so either process can invalidate the job and both can
//! perform a bounded coordinated shutdown. This is deliberately separate from
//! communicator bootstrap: rendezvous proves process identity and lifecycle,
//! while a transport still proves its own plan and communicator identity.
//!
//! Split by concern per `docs/CONVENTIONS.md`: `rank` is the type-level
//! two-rank vocabulary (`StaticTwoRank`); `identity` is `RunId` and the
//! agreed `DistributedIdentity`; `state` is the fail-stop lifecycle state
//! machine (`DistributedContextState`, `ContextFailure`,
//! `DistributedContextHandle`); `error` is the failure vocabulary shared by
//! every other module; `lifecycle` is the `DistributedContext` type and its
//! abort/wait_for_peer/shutdown surface; `rendezvous` is the launcher-facing
//! config/plan vocabulary plus the entry points that turn a config into a
//! context; `bootstrap` is the TCP accept/connect protocol driver; `wire` is
//! the startup/control message encoding. `rendezvous`, `bootstrap`, and
//! `wire` are `std`-only and gated at the module declaration, not per item.

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

pub(super) const MAX_RUN_ID_BYTES: usize = 128;
pub(super) const STATE_ACTIVE: u8 = 0;
pub(super) const STATE_SHUTTING_DOWN: u8 = 1;
pub(super) const STATE_SHUTDOWN: u8 = 2;
pub(super) const STATE_FAILED: u8 = 3;

#[cfg(feature = "std")]
mod bootstrap;
mod error;
mod identity;
mod lifecycle;
mod rank;
#[cfg(feature = "std")]
mod rendezvous;
mod state;
#[cfg(feature = "std")]
mod wire;

pub use error::ContextError;
pub use identity::{DistributedIdentity, RunId};
pub use lifecycle::DistributedContext;
pub use rank::StaticTwoRank;
#[cfg(feature = "std")]
pub use rendezvous::{
    DynRendezvousConfig, RankLaunch, RendezvousEndpoint, StaticRendezvousConfig, TwoRankLaunchPlan,
};
pub use state::{ContextFailure, DistributedContextHandle, DistributedContextState};
