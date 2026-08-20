//! Authoritative native-backend capability registrations.
//!
//! Split by concern per `docs/CONVENTIONS.md`: `constants` is the shared
//! dtype/layout/math-mode vocabulary; `declarations` is the four backends'
//! `*_descriptor_operations!` macros, each backend's own declaration of the
//! identities it advertises; `rules` is `descriptor_capability_rules!`, the
//! callback that turns a backend's per-group declaration into `CapabilityRule`
//! rows, plus the row constructors and rank-bound helpers it alone uses;
//! `tables` is the four `pub static ..._CAPABILITIES` tables built by feeding
//! `declarations` into `rules`; `query` is the public lookup surface
//! (`registry`, `support`, `coverage_report`) that reads them. `tests` cross-
//! references all four tables against `OPERATION_CATALOG`.
//!
//! The tables themselves carry no `#[cfg]`: a capability claim is data, and
//! `registry`/`coverage_report` report every backend's regardless of which
//! are compiled in. Only the four `*_descriptor_operations!` re-exports below
//! are feature-gated, since only the executor module that checks a macro
//! against its own `Execute<op::...>` implementations needs the gate;
//! `tables` reaches every macro through the ungated path in `declarations`.

mod constants;
mod declarations;
mod query;
mod rules;
mod tables;
#[cfg(test)]
mod tests;

pub use query::{BackendCoverageRow, coverage_report, registry, support};
pub use tables::{CPU_CAPABILITIES, CUDA_CAPABILITIES, METAL_CAPABILITIES, WGPU_CAPABILITIES};

#[cfg(feature = "cpu")]
pub(crate) use declarations::cpu_descriptor_operations;
#[cfg(feature = "cuda")]
pub(crate) use declarations::cuda_descriptor_operations;
#[cfg(feature = "metal")]
pub(crate) use declarations::metal_descriptor_operations;
#[cfg(feature = "wgpu")]
pub(crate) use declarations::wgpu_descriptor_operations;
