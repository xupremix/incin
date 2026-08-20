//! Topology discovery and device probing for NCCL transport.

use cudarc::driver::CudaContext;
use incin_core::dist::DistributedContext;
use incin_core::dist::mesh::{
    DeviceIdentity, LinkClass, ProcessLayout, TopologyProbe, TransportVersion,
};
use incin_core::tensor::device::DeviceId;

use crate::dist::nccl::config::{TwoRankBootstrapConfig, WORLD, bootstrap_from_context};
use crate::dist::nccl::error::{NcclTransportError, catch_nccl_panic, nccl_error};
use crate::dist::nccl::wire::{exchange_topology, format_cuda_uuid};

/// Physical topology shared by both process-per-rank NCCL participants.
///
/// A launcher gathers one [`probe_local_cuda_identity`](Self::probe_local_cuda_identity)
/// result from each host, preserves rank order, and gives the same pair back to
/// both processes. Both ranks then derive one [`MeshId`](incin_core::dist::mesh::MeshId) even when each host
/// exposes its local GPU as CUDA ordinal zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NcclTopology {
    identities: [DeviceIdentity; WORLD],
    rank: usize,
    transport: TransportVersion,
}

impl NcclTopology {
    /// Discover both hosts' physical identities over the bootstrap socket.
    ///
    /// This is the first of two bounded TCP sessions: discovery supplies the
    /// shared topology needed to bind a mesh and build a plan; later
    /// [`NcclTransport::connect`](crate::dist::nccl::NcclTransport::connect) exchanges that plan and the NCCL unique id.
    pub fn discover(
        config: TwoRankBootstrapConfig,
        device_ordinal: usize,
    ) -> Result<Self, NcclTransportError> {
        if config.timeout.is_zero() {
            return Err(NcclTransportError::InvalidTimeout);
        }
        let identity = Self::probe_local_cuda_identity(config.rank(), device_ordinal)?;
        let transport = Self::installed_transport_version()?;
        exchange_topology(config, identity, transport)
    }

    /// Discover topology from an agreed launcher context.
    ///
    /// A failure invalidates every clone and backend handle derived from the
    /// context, matching the same fail-stop rule as communicator creation.
    pub fn discover_context<M, R>(
        context: &DistributedContext<M, R>,
    ) -> Result<Self, NcclTransportError> {
        context.ensure_active()?;
        let config = bootstrap_from_context(context);
        let handle = context.handle();
        match Self::discover(config, context.local_cuda_device()) {
            Ok(topology) => Ok(topology),
            Err(error) => {
                handle.invalidate();
                Err(error)
            }
        }
    }

    /// Build the topology after the launcher has exchanged both identities.
    pub fn new(
        identities: [DeviceIdentity; WORLD],
        rank: usize,
        transport: TransportVersion,
    ) -> Result<Self, NcclTransportError> {
        if rank >= WORLD {
            return Err(NcclTransportError::LocalRank { rank, world: WORLD });
        }
        Ok(Self {
            identities,
            rank,
            transport,
        })
    }

    /// Query this process's stable CUDA UUID and compute capability.
    ///
    /// `rank` becomes the logical mesh ordinal. `device_ordinal` is local to
    /// this host after its CUDA visibility mask is applied.
    pub fn probe_local_cuda_identity(
        rank: usize,
        device_ordinal: usize,
    ) -> Result<DeviceIdentity, NcclTransportError> {
        if rank >= WORLD {
            return Err(NcclTransportError::LocalRank { rank, world: WORLD });
        }
        let context = CudaContext::new(device_ordinal)
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        let uuid = context
            .uuid()
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        let (major, minor) = context
            .compute_capability()
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        Ok(DeviceIdentity::new(
            DeviceId::cuda(rank),
            format_cuda_uuid(uuid.bytes),
            format!("sm_{major}{minor}"),
        ))
    }

    /// Query the dynamically loaded NCCL library version.
    pub fn installed_transport_version() -> Result<TransportVersion, NcclTransportError> {
        let encoded = catch_nccl_panic("query version", cudarc::nccl::result::get_nccl_version)?
            .map_err(nccl_error)?;
        let encoded =
            u32::try_from(encoded).map_err(|_| NcclTransportError::InvalidNcclVersion(encoded))?;
        Ok(TransportVersion::new(
            "nccl".to_string(),
            encoded / 10_000,
            (encoded / 100) % 100,
            encoded % 100,
        ))
    }
}

impl TopologyProbe for NcclTopology {
    fn identify(&self, device: DeviceId) -> Option<DeviceIdentity> {
        self.identities
            .iter()
            .find(|identity| identity.device() == device)
            .cloned()
    }

    fn link(&self, from: DeviceId, to: DeviceId) -> LinkClass {
        let known = |device| {
            self.identities
                .iter()
                .any(|identity| identity.device() == device)
        };
        if !known(from) || !known(to) {
            LinkClass::Unreachable
        } else if from == to {
            LinkClass::SameDevice
        } else {
            LinkClass::Network
        }
    }

    fn transport(&self) -> TransportVersion {
        self.transport.clone()
    }

    fn layout(&self) -> ProcessLayout {
        ProcessLayout::ProcessPerRank {
            rank: self.rank,
            world: WORLD,
        }
    }
}
