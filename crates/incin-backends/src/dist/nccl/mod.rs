//! Two-rank, process-per-rank NCCL transport.
//!
//! The transport deliberately has a rank-local API: unlike the deterministic
//! reference backend, one process owns one input buffer and NCCL supplies the
//! other rank over the network. A fixed-size TCP bootstrap exchanges the NCCL
//! unique id and the [`PlanSummary`](incin_core::dist::PlanSummary) before either process initializes its
//! communicator.

pub(crate) mod buffer;
pub(crate) mod config;
pub(crate) mod error;
pub(crate) mod topology;
pub(crate) mod transport;
pub(crate) mod wire;

#[cfg(test)]
mod tests;

pub use buffer::{NcclBuffer, NcclEvent};
pub use config::{BootstrapRole, TwoRankBootstrapConfig};
pub use error::NcclTransportError;
pub use topology::NcclTopology;
pub use transport::NcclTransport;
