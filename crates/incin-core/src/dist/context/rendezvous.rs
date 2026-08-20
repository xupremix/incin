//! The launcher-facing rendezvous configuration and plan vocabulary, plus
//! the entry points that turn a config into a live context.
//!
//! This module is gated at the declaration site in `mod.rs`, so its items do
//! not repeat the `std` feature attribute individually.

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::marker::PhantomData;
use std::net::SocketAddr;
use std::process::Command;
use std::time::Duration;

use typenum::{U0, U1, U2};

use crate::dist::mesh::ValidMesh;
use crate::shapes::Dyn;

use super::bootstrap::{parse_env, read_env, rendezvous, validate_dyn_launch, validate_timeout};
use super::error::ContextError;
use super::identity::RunId;
use super::lifecycle::DistributedContext;
use super::rank::StaticTwoRank;
use super::{
    LOCAL_CUDA_DEVICE_ENV, RANK_ENV, RENDEZVOUS_ADDR_ENV, RENDEZVOUS_TIMEOUT_MS_ENV, RUN_ID_ENV,
    TWO_RANK_WORLD, WORLD_SIZE_ENV,
};

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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RendezvousEndpoint {
    /// Rank zero's bind address.
    Root { bind: SocketAddr },
    /// Rank zero's address as reachable from rank one.
    Peer { root: SocketAddr },
}

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
#[derive(Debug, Clone)]
pub struct StaticRendezvousConfig<R> {
    run_id: RunId,
    endpoint: RendezvousEndpoint,
    local_cuda_device: usize,
    timeout: Duration,
    marker: PhantomData<R>,
}

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
#[derive(Debug, Clone)]
pub struct DynRendezvousConfig {
    run_id: RunId,
    endpoint: RendezvousEndpoint,
    rank: usize,
    world: usize,
    local_cuda_device: usize,
    timeout: Duration,
}

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
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RankLaunch {
    run_id: RunId,
    root: SocketAddr,
    rank: usize,
    world: usize,
    local_cuda_device: usize,
    timeout_ms: u64,
}

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
#[derive(Debug, Clone)]
pub struct TwoRankLaunchPlan<M = Dyn> {
    run_id: RunId,
    root: SocketAddr,
    local_cuda_devices: [usize; TWO_RANK_WORLD],
    timeout: Duration,
    marker: PhantomData<M>,
}

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
