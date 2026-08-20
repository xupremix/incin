//! The physical two-rank topology the hybrid planner compares strategies
//! against.

use super::*;

/// A physical two-rank topology shared by DP=2, TP=2, and PP=2 candidates.
///
/// The topology intentionally contains no logical mesh degrees. The hybrid
/// planner compares several logical interpretations of the same two devices;
/// carrying the `MeshId` of whichever interpretation happened to be bound
/// first would bias that comparison.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TwoRankPlanningTopology {
    pub(super) fingerprint: u64,
    pub(super) link: LinkClass,
    pub(super) transport: String,
    pub(super) process_layout: ProcessLayout,
}

impl TwoRankPlanningTopology {
    /// Project a statically two-rank bound mesh into the planner's physical
    /// topology.
    ///
    /// A mesh whose type-level world is not exactly two does not satisfy this
    /// function's bound and therefore fails to compile.
    pub fn from_static_mesh<M>(mesh: &DeviceMesh<M>) -> Result<Self, HybridPlanError>
    where
        M: ValidMesh<World = U2>,
    {
        Self::from_fingerprint(mesh.fingerprint())
    }

    /// Validate a runtime-discovered topology for the `Dyn` planning path.
    pub fn from_fingerprint(fingerprint: &TopologyFingerprint) -> Result<Self, HybridPlanError> {
        if fingerprint.devices().len() != 2 {
            return Err(HybridPlanError::TopologyWorld {
                expected: 2,
                found: fingerprint.devices().len(),
            });
        }

        if let ProcessLayout::ProcessPerRank { world, .. } = fingerprint.layout()
            && *world != 2
        {
            return Err(HybridPlanError::ProcessWorld {
                expected: 2,
                found: *world,
            });
        }

        let forward = fingerprint
            .links()
            .iter()
            .find_map(|&(from, to, class)| (from == 0 && to == 1).then_some(class))
            .ok_or(HybridPlanError::MissingLink {
                from_rank: 0,
                to_rank: 1,
            })?;
        let backward = fingerprint
            .links()
            .iter()
            .find_map(|&(from, to, class)| (from == 1 && to == 0).then_some(class))
            .ok_or(HybridPlanError::MissingLink {
                from_rank: 1,
                to_rank: 0,
            })?;
        if !forward.reaches() {
            return Err(HybridPlanError::UnreachableLink {
                from_rank: 0,
                to_rank: 1,
            });
        }
        if !backward.reaches() {
            return Err(HybridPlanError::UnreachableLink {
                from_rank: 1,
                to_rank: 0,
            });
        }

        Ok(Self {
            fingerprint: fingerprint.digest(),
            link: core::cmp::max(forward, backward),
            transport: fingerprint.transport().library().to_owned(),
            process_layout: fingerprint.layout().clone(),
        })
    }

    /// Stable physical-topology identity used by the report.
    #[must_use]
    pub const fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    /// Least direct of the two ordered links between ranks.
    #[must_use]
    pub const fn link(&self) -> LinkClass {
        self.link
    }

    /// Communication library reported by topology discovery.
    #[must_use]
    pub fn transport(&self) -> &str {
        &self.transport
    }

    /// Process layout included in the topology assumption.
    #[must_use]
    pub const fn process_layout(&self) -> &ProcessLayout {
        &self.process_layout
    }
}
