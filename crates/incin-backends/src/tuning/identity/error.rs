//! Failure vocabulary and canonical-field validation for tuning identities.

use alloc::string::{String, ToString};

const MAX_IDENTITY_FIELD_BYTES: usize = 256;

/// A failure to construct or project a stable identity.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum IdentityError {
    /// A required field was empty.
    #[error("tuning identity field `{field}` must not be empty")]
    EmptyField {
        /// The field name.
        field: &'static str,
    },
    /// A field exceeded the persistent-format bound.
    #[error("tuning identity field `{field}` is {actual} bytes; maximum is {maximum}")]
    FieldTooLong {
        /// The field name.
        field: &'static str,
        /// Supplied byte length.
        actual: usize,
        /// Maximum byte length.
        maximum: usize,
    },
    /// A field contained whitespace/control data which is not canonical.
    #[error("tuning identity field `{field}` is not canonical printable ASCII")]
    NonCanonicalField {
        /// The field name.
        field: &'static str,
    },
    /// A future runtime backend is not yet understood by this identity schema.
    #[error("runtime backend is not supported by tuning identity schema v1")]
    UnsupportedBackend,
    /// A `Dyn` identity cannot be projected to the requested static backend.
    #[error("tuning identity backend mismatch: expected {expected}, found {actual}")]
    BackendMismatch {
        /// Static backend requested by the caller.
        expected: &'static str,
        /// Runtime backend found in the identity.
        actual: &'static str,
    },
    /// A dynamic topology declared an empty world.
    #[error("tuning topology world size must be nonzero")]
    ZeroWorld,
    /// The number of rank identities did not match the declared world.
    #[error("tuning topology declares world {world} but contains {devices} rank devices")]
    WorldMismatch {
        /// Declared static or dynamic world.
        world: usize,
        /// Number of device records.
        devices: usize,
    },
    /// A dynamic topology had a different world from the requested static one.
    #[error("cannot project dynamic topology world {actual} to static world {expected}")]
    StaticWorldMismatch {
        /// Type-level world requested by the caller.
        expected: usize,
        /// Runtime world stored in the topology.
        actual: usize,
    },
    /// Two ranks named the same stable physical device.
    #[error(
        "tuning topology aliases physical device `{persistent_id}` at ranks {first_rank} and {second_rank}"
    )]
    AliasedDevice {
        /// Vendor-persistent identifier.
        persistent_id: String,
        /// First rank using the device.
        first_rank: usize,
        /// Second rank using the same device.
        second_rank: usize,
    },
    /// A link referred to a rank not in the topology.
    #[error("tuning topology link {from}->{to} is outside world {world}")]
    LinkOutOfRange {
        /// Link source rank.
        from: usize,
        /// Link destination rank.
        to: usize,
        /// Topology world.
        world: usize,
    },
    /// A link from a rank to itself is not a communication edge.
    #[error("tuning topology link {rank}->{rank} is a self link")]
    SelfLink {
        /// The repeated rank.
        rank: usize,
    },
    /// The same directed edge was recorded more than once.
    #[error("tuning topology contains duplicate link {from}->{to}")]
    DuplicateLink {
        /// Link source rank.
        from: usize,
        /// Link destination rank.
        to: usize,
    },
    /// The process layout does not cover exactly the declared ranks.
    #[error("process layout {processes}x{ranks_per_process} does not cover topology world {world}")]
    ProcessLayoutMismatch {
        /// Number of processes.
        processes: usize,
        /// Ranks assigned to each process.
        ranks_per_process: usize,
        /// Topology world.
        world: usize,
    },
    /// A CUDA runtime query failed.
    #[error("failed to query CUDA {component}: {message}")]
    CudaQuery {
        /// Driver field being queried.
        component: &'static str,
        /// Driver error.
        message: String,
    },
}

pub(super) fn checked_field(
    field: &'static str,
    value: &str,
) -> core::result::Result<String, IdentityError> {
    if value.is_empty() {
        return Err(IdentityError::EmptyField { field });
    }
    if value.len() > MAX_IDENTITY_FIELD_BYTES {
        return Err(IdentityError::FieldTooLong {
            field,
            actual: value.len(),
            maximum: MAX_IDENTITY_FIELD_BYTES,
        });
    }
    if value.trim() != value
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_graphic() || byte == b' ')
    {
        return Err(IdentityError::NonCanonicalField { field });
    }
    Ok(value.to_string())
}
