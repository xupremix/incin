//! Cross-rank plan-summary agreement before any collective launches.

use super::*;

/// Compact plan identity exchanged by every rank before launch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlanSummary {
    pub(super) mesh: MeshId,
    pub(super) hash: u64,
    pub(super) collectives: usize,
}

impl PlanSummary {
    /// Rebuild a summary received from another process.
    ///
    /// A reconstructed summary is only data. [`preflight`] is what compares
    /// all ranks and mints the [`AgreedPlan`] proof used by a transport.
    #[must_use]
    pub const fn from_parts(mesh: MeshId, hash: u64, collectives: usize) -> Self {
        Self {
            mesh,
            hash,
            collectives,
        }
    }

    /// Mesh identity.
    #[must_use]
    pub const fn mesh_id(self) -> MeshId {
        self.mesh
    }

    /// Stable descriptor hash.
    #[must_use]
    pub const fn hash(self) -> u64 {
        self.hash
    }

    /// Number of collective launches.
    #[must_use]
    pub const fn collective_count(self) -> usize {
        self.collectives
    }
}

/// Sealed result of all-rank plan agreement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AgreedPlan {
    summary: PlanSummary,
    ranks: usize,
}

impl AgreedPlan {
    /// Agreed plan summary.
    #[must_use]
    pub const fn summary(self) -> PlanSummary {
        self.summary
    }

    /// Number of ranks participating in preflight.
    #[must_use]
    pub const fn ranks(self) -> usize {
        self.ranks
    }
}

/// Compare mesh, count, and hash before any collective launch.
pub fn preflight(
    expected_ranks: usize,
    summaries: &[PlanSummary],
) -> Result<AgreedPlan, PlanError> {
    if expected_ranks == 0 {
        return Err(PlanError::EmptyPreflight);
    }
    if summaries.len() != expected_ranks {
        return Err(PlanError::PreflightRankCount {
            expected: expected_ranks,
            found: summaries.len(),
        });
    }
    let expected = summaries[0];
    for (rank, &found) in summaries.iter().enumerate().skip(1) {
        if found.mesh != expected.mesh {
            return Err(PlanError::MeshMismatch {
                rank,
                expected: expected.mesh,
                found: found.mesh,
            });
        }
        if found.collectives != expected.collectives {
            return Err(PlanError::CollectiveCountMismatch {
                rank,
                expected: expected.collectives,
                found: found.collectives,
            });
        }
        if found.hash != expected.hash {
            return Err(PlanError::PlanHashMismatch {
                rank,
                expected: expected.hash,
                found: found.hash,
            });
        }
    }
    Ok(AgreedPlan {
        summary: expected,
        ranks: expected_ranks,
    })
}
