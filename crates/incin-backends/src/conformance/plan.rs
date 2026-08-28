//! Enumerating what a backend advertises, one executable tuple at a time.
//!
//! `docs/capabilities.md` renders a [`CapabilityRule`] as a single row: an
//! operation, a dtype set, a layout set, a rank range, a training flag. A row
//! is not one claim. It is the product of those sets, and every point in that
//! product is a separate promise the backend made. A harness that runs one
//! representative point per row checks the operation and nothing else, which
//! is already proved at compile time by the `Execute` obligation
//! `crates/incin-backends/src/cpu/canonical/mod.rs` installs.
//!
//! So the unit here is the tuple, not the row. Expanding the product is what
//! turns "advertises `f32` through `f64` at ranks one to four" from prose into
//! something that can fail.

use alloc::vec::Vec;

use incin_core::exec::{
    CapabilityQuery, CapabilityRule, LayoutClass, MathMode, OperationIdentity, catalog_entry,
};
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::device::DeviceKind;
use incin_core::tensor::dtype::DTypeDescriptor;

/// Highest rank the harness will materialize.
///
/// A rule may say `usize::MAX`, which is an honest claim about a kernel that
/// walks its operand generically and a dishonest amount of memory to allocate.
/// Four is where the boundary stops being interesting: it is the rank of an
/// `[N, C, H, W]` activation, it is above every special case in the shape
/// rules, and a kernel correct at four is not newly wrong at nine for reasons
/// this harness could observe.
pub const RANK_CAP: usize = 4;

/// One point in the product a capability row describes.
///
/// Carries the whole query rather than a subset because the query is what the
/// registry answers, and reconstructing it later from parts is how a harness
/// ends up checking a tuple the backend was never asked about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AdvertisedTuple {
    /// Exact catalog operation the row describes.
    pub operation: OperationKind,
    /// The one dtype this point claims, drawn from the row's set.
    pub dtype: DTypeDescriptor,
    /// The one layout class this point claims, drawn from the row's set.
    pub layout: LayoutClass,
    /// The rank this point claims.
    pub rank: usize,
    /// Whether the row claims training-mode execution.
    pub training: bool,
    /// The one math mode this point claims, drawn from the row's set.
    pub math_mode: MathMode,
}

impl AdvertisedTuple {
    /// The capability query that admits exactly this tuple.
    #[must_use]
    pub const fn query(&self) -> CapabilityQuery {
        CapabilityQuery {
            operation: OperationIdentity::Builtin(self.operation),
            dtype: self.dtype,
            layout: self.layout,
            rank: self.rank,
            training: self.training,
            math_mode: self.math_mode,
        }
    }

    /// A stable one-line identity for reports and failure messages.
    #[must_use]
    pub fn label(&self) -> alloc::string::String {
        alloc::format!(
            "{} [{:?}, {:?}, rank {}]",
            self.operation,
            self.dtype.name(),
            self.layout,
            self.rank
        )
    }
}

/// The ranks worth executing for one rule.
///
/// Both ends, and the catalog's own floor when the row declares one below it.
/// The interior of a rank range is where a kernel is least likely to be wrong,
/// because a kernel that handles rank two and rank four handles rank three by
/// the same code path; the ends are where a squeeze, a broadcast, or an axis
/// normalization runs out of dimensions. The precedent is in
/// `docs/book/src/backend_authoring.md`, which records that executing boundary
/// cases is what found rows advertising ranks their kernels refused.
///
/// The catalog floor is a third end rather than an interior point, because a
/// row whose floor sits below it is a union row and the two numbers are
/// boundaries of two different operands.
fn boundary_ranks(rule: &CapabilityRule) -> Vec<usize> {
    let entry = catalog_entry(rule.operation);
    let catalog_max = entry
        .map_or(RANK_CAP, |entry| *entry.accepted_ranks.end())
        .min(RANK_CAP);
    let catalog_floor = entry.map_or(0, |entry| *entry.accepted_ranks.start());
    let low = rule.min_rank;
    let high = rule.max_rank.min(catalog_max).max(low);

    let mut ranks = alloc::vec![low];
    // A union row states the loosest rank across all of its operands, so its
    // floor can speak for a bias vector while the primary operand's own floor
    // sits above it. `conv2d` declares one and accepts three, which puts the
    // rank where the activation stops having a channel axis in the *interior*
    // of the range, exactly where boundary enumeration does not look. The
    // catalog's floor is that second boundary, and adding it is what makes the
    // rank-three convolution reachable at all.
    if catalog_floor > low && catalog_floor < high {
        ranks.push(catalog_floor);
    }
    if high > low {
        ranks.push(high);
    }
    ranks
}

/// Every tuple `device`'s registry advertises, in table order.
///
/// Table order rather than sorted order so a report reads in the same sequence
/// as the declaration a reader would go check.
#[must_use]
pub fn advertised_tuples(device: DeviceKind) -> Vec<AdvertisedTuple> {
    let registry = crate::capability::registry(device);
    let mut tuples = Vec::new();

    for rule in registry.registrations() {
        for rank in boundary_ranks(rule) {
            for dtype in rule.dtypes {
                for layout in rule.layouts {
                    for math_mode in rule.math_modes {
                        tuples.push(AdvertisedTuple {
                            operation: rule.operation,
                            dtype: *dtype,
                            layout: *layout,
                            rank,
                            training: rule.training,
                            math_mode: *math_mode,
                        });
                    }
                }
            }
        }
    }

    tuples
}
