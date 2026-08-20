//! Stable rank mapping, links, transport, and process layout.

use alloc::{string::String, vec::Vec};
use core::{fmt, marker::PhantomData};

use incin_core::{
    shapes::Dyn,
    typenum::{NonZero, Unsigned},
};

use super::device::DeviceFingerprint;
use super::error::{IdentityError, checked_field};
use super::primitives::{Digest, IDENTITY_SCHEMA, SoftwareVersion};

/// A type-level, nonzero topology world.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StaticWorld<N: Unsigned + NonZero>(PhantomData<N>);

/// A directed communication link included in a topology identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TopologyLink {
    from: usize,
    to: usize,
    class: String,
}

impl TopologyLink {
    /// Records a directed link class.
    pub fn new(from: usize, to: usize, class: &str) -> core::result::Result<Self, IdentityError> {
        Ok(Self {
            from,
            to,
            class: checked_field("link_class", class)?,
        })
    }

    /// Source rank.
    #[must_use]
    pub const fn from(&self) -> usize {
        self.from
    }

    /// Destination rank.
    #[must_use]
    pub const fn to(&self) -> usize {
        self.to
    }

    /// Stable link class.
    #[must_use]
    pub fn class(&self) -> &str {
        &self.class
    }
}

/// Communication-library identity.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TransportFingerprint {
    library: String,
    version: SoftwareVersion,
}

impl TransportFingerprint {
    /// Constructs a transport identity.
    pub fn new(
        library: &str,
        version: SoftwareVersion,
    ) -> core::result::Result<Self, IdentityError> {
        Ok(Self {
            library: checked_field("transport_library", library)?,
            version,
        })
    }

    /// Library name.
    #[must_use]
    pub fn library(&self) -> &str {
        &self.library
    }

    /// Library version.
    #[must_use]
    pub const fn version(&self) -> SoftwareVersion {
        self.version
    }
}

/// Rank-to-process layout, excluding the observing process's local rank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProcessLayoutFingerprint {
    processes: usize,
    ranks_per_process: usize,
}

impl ProcessLayoutFingerprint {
    /// Constructs a uniform process layout.
    #[must_use]
    pub const fn new(processes: usize, ranks_per_process: usize) -> Self {
        Self {
            processes,
            ranks_per_process,
        }
    }

    /// Number of processes.
    #[must_use]
    pub const fn processes(self) -> usize {
        self.processes
    }

    /// Number of ranks driven by each process.
    #[must_use]
    pub const fn ranks_per_process(self) -> usize {
        self.ranks_per_process
    }
}

/// Stable rank mapping, links, transport, and process layout.
///
/// `W = StaticWorld<N>` carries a nonzero world at compile time. `W = Dyn`
/// stores a runtime world and applies the same cardinality and alias checks.
pub struct TuningTopologyFingerprint<W = Dyn> {
    world: usize,
    devices: Vec<DeviceFingerprint<Dyn>>,
    links: Vec<TopologyLink>,
    transport: TransportFingerprint,
    layout: ProcessLayoutFingerprint,
    marker: PhantomData<fn() -> W>,
}

impl<N> TuningTopologyFingerprint<StaticWorld<N>>
where
    N: Unsigned + NonZero + 'static,
{
    /// Constructs a topology whose world is carried by `N`.
    pub fn new(
        devices: Vec<DeviceFingerprint<Dyn>>,
        links: Vec<TopologyLink>,
        transport: TransportFingerprint,
        layout: ProcessLayoutFingerprint,
    ) -> core::result::Result<Self, IdentityError> {
        Self::from_parts(N::USIZE, devices, links, transport, layout)
    }
}

impl TuningTopologyFingerprint<Dyn> {
    /// Constructs a runtime-selected topology.
    pub fn new_dyn(
        world: usize,
        devices: Vec<DeviceFingerprint<Dyn>>,
        links: Vec<TopologyLink>,
        transport: TransportFingerprint,
        layout: ProcessLayoutFingerprint,
    ) -> core::result::Result<Self, IdentityError> {
        Self::from_parts(world, devices, links, transport, layout)
    }

    /// Projects a runtime topology to a statically known nonzero world.
    pub fn try_into_static<N>(
        self,
    ) -> core::result::Result<TuningTopologyFingerprint<StaticWorld<N>>, IdentityError>
    where
        N: Unsigned + NonZero + 'static,
    {
        if self.world != N::USIZE {
            return Err(IdentityError::StaticWorldMismatch {
                expected: N::USIZE,
                actual: self.world,
            });
        }
        Ok(TuningTopologyFingerprint {
            world: self.world,
            devices: self.devices,
            links: self.links,
            transport: self.transport,
            layout: self.layout,
            marker: PhantomData,
        })
    }
}

impl<W> TuningTopologyFingerprint<W> {
    /// Erases a static or dynamic world marker while retaining the checked
    /// runtime value.
    #[must_use]
    pub fn erase(self) -> TuningTopologyFingerprint<Dyn> {
        TuningTopologyFingerprint {
            world: self.world,
            devices: self.devices,
            links: self.links,
            transport: self.transport,
            layout: self.layout,
            marker: PhantomData,
        }
    }

    fn from_parts(
        world: usize,
        devices: Vec<DeviceFingerprint<Dyn>>,
        mut links: Vec<TopologyLink>,
        transport: TransportFingerprint,
        layout: ProcessLayoutFingerprint,
    ) -> core::result::Result<Self, IdentityError> {
        if world == 0 {
            return Err(IdentityError::ZeroWorld);
        }
        if devices.len() != world {
            return Err(IdentityError::WorldMismatch {
                world,
                devices: devices.len(),
            });
        }
        for second_rank in 0..devices.len() {
            if let Some(first_rank) = (0..second_rank).find(|&first_rank| {
                devices[first_rank].physical_key() == devices[second_rank].physical_key()
            }) {
                return Err(IdentityError::AliasedDevice {
                    persistent_id: devices[second_rank].persistent_id.clone(),
                    first_rank,
                    second_rank,
                });
            }
        }
        links.sort();
        for (index, link) in links.iter().enumerate() {
            if link.from >= world || link.to >= world {
                return Err(IdentityError::LinkOutOfRange {
                    from: link.from,
                    to: link.to,
                    world,
                });
            }
            if link.from == link.to {
                return Err(IdentityError::SelfLink { rank: link.from });
            }
            if index > 0 && links[index - 1].from == link.from && links[index - 1].to == link.to {
                return Err(IdentityError::DuplicateLink {
                    from: link.from,
                    to: link.to,
                });
            }
        }
        let covered = layout
            .processes
            .checked_mul(layout.ranks_per_process)
            .filter(|&covered| covered == world);
        if covered.is_none() {
            return Err(IdentityError::ProcessLayoutMismatch {
                processes: layout.processes,
                ranks_per_process: layout.ranks_per_process,
                world,
            });
        }
        Ok(Self {
            world,
            devices,
            links,
            transport,
            layout,
            marker: PhantomData,
        })
    }

    /// Number of ranks in the topology.
    #[must_use]
    pub const fn world(&self) -> usize {
        self.world
    }

    /// Stable identity bound to each rank, in rank order.
    #[must_use]
    pub fn devices(&self) -> &[DeviceFingerprint<Dyn>] {
        &self.devices
    }

    /// Directed links in canonical order.
    #[must_use]
    pub fn links(&self) -> &[TopologyLink] {
        &self.links
    }

    /// Communication-library identity.
    #[must_use]
    pub const fn transport(&self) -> &TransportFingerprint {
        &self.transport
    }

    /// Rank-to-process layout.
    #[must_use]
    pub const fn layout(&self) -> ProcessLayoutFingerprint {
        self.layout
    }

    /// Stable digest excluding process-local ordinals and observing rank.
    #[must_use]
    pub fn digest(&self) -> u64 {
        let mut digest = Digest::new()
            .field(IDENTITY_SCHEMA)
            .field(b"topology")
            .number(self.world as u64);
        for device in &self.devices {
            digest = digest.number(device.digest());
        }
        for link in &self.links {
            digest = digest
                .number(link.from as u64)
                .number(link.to as u64)
                .text(&link.class);
        }
        digest
            .text(&self.transport.library)
            .version(self.transport.version)
            .number(self.layout.processes as u64)
            .number(self.layout.ranks_per_process as u64)
            .finish()
    }
}

impl<W> Clone for TuningTopologyFingerprint<W> {
    fn clone(&self) -> Self {
        Self {
            world: self.world,
            devices: self.devices.clone(),
            links: self.links.clone(),
            transport: self.transport.clone(),
            layout: self.layout,
            marker: PhantomData,
        }
    }
}

impl<W> fmt::Debug for TuningTopologyFingerprint<W> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TuningTopologyFingerprint")
            .field("world", &self.world)
            .field("devices", &self.devices)
            .field("links", &self.links)
            .field("transport", &self.transport)
            .field("layout", &self.layout)
            .finish()
    }
}

impl<W> PartialEq for TuningTopologyFingerprint<W> {
    fn eq(&self, other: &Self) -> bool {
        self.world == other.world
            && self.devices == other.devices
            && self.links == other.links
            && self.transport == other.transport
            && self.layout == other.layout
    }
}

impl<W> Eq for TuningTopologyFingerprint<W> {}
