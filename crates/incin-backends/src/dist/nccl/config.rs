//! TCP bootstrap configuration and roles for two-rank NCCL transport.

use std::net::SocketAddr;
use std::time::Duration;

use incin_core::dist::{DistributedContext, RendezvousEndpoint};

pub(crate) const WORLD: usize = 2;
pub(crate) const UNIQUE_ID_BYTES: usize = 128;
pub(crate) const MAGIC: [u8; 8] = *b"INCINN01";
pub(crate) const WIRE_BYTES: usize = 8 + 1 + 1 + 6 + 8 + 8 + 8 + UNIQUE_ID_BYTES;
pub(crate) const TOPOLOGY_MAGIC: [u8; 8] = *b"INCINT01";
pub(crate) const PERSISTENT_BYTES: usize = 64;
pub(crate) const ARCHITECTURE_BYTES: usize = 32;
pub(crate) const LIBRARY_BYTES: usize = 16;
pub(crate) const TOPOLOGY_WIRE_BYTES: usize = 8
    + 1
    + 1
    + 6
    + 4
    + 4
    + 4
    + 2
    + 2
    + 2
    + 2
    + PERSISTENT_BYTES
    + ARCHITECTURE_BYTES
    + LIBRARY_BYTES;

/// Which side of the two-rank TCP bootstrap this process owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootstrapRole {
    /// Rank zero listens at this address and creates the NCCL unique id.
    Root {
        /// Address rank zero listens on.
        bind: SocketAddr,
    },
    /// Rank one connects to rank zero at this address.
    Peer {
        /// Root address peers connect to.
        root: SocketAddr,
    },
}

impl BootstrapRole {
    pub(crate) const fn rank(self) -> usize {
        match self {
            Self::Root { .. } => 0,
            Self::Peer { .. } => 1,
        }
    }
}

/// TCP bootstrap settings for exactly two network-accessible CUDA ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TwoRankBootstrapConfig {
    pub(crate) role: BootstrapRole,
    pub(crate) timeout: Duration,
}

impl TwoRankBootstrapConfig {
    /// Configure rank zero.
    #[must_use]
    pub const fn root(bind: SocketAddr, timeout: Duration) -> Self {
        Self {
            role: BootstrapRole::Root { bind },
            timeout,
        }
    }

    /// Configure rank one.
    #[must_use]
    pub const fn peer(root: SocketAddr, timeout: Duration) -> Self {
        Self {
            role: BootstrapRole::Peer { root },
            timeout,
        }
    }

    /// This process's rank.
    #[must_use]
    pub const fn rank(self) -> usize {
        self.role.rank()
    }

    /// Startup and socket I/O deadline.
    #[must_use]
    pub const fn timeout(self) -> Duration {
        self.timeout
    }

    /// Root/peer role and network address.
    #[must_use]
    pub const fn role(self) -> BootstrapRole {
        self.role
    }
}

pub(crate) fn bootstrap_from_context<M, R>(
    context: &DistributedContext<M, R>,
) -> TwoRankBootstrapConfig {
    match context.endpoint() {
        RendezvousEndpoint::Root { bind } => TwoRankBootstrapConfig::root(bind, context.timeout()),
        RendezvousEndpoint::Peer { root } => TwoRankBootstrapConfig::peer(root, context.timeout()),
    }
}
