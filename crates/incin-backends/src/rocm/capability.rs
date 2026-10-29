//! ROCm capability registration: the exact empty table.
//!
//! This slice implies no kernels, so it advertises none: every
//! [`CapabilityQuery`](incin_core::backend_authoring::CapabilityQuery) against
//! [`DeviceKind::Rocm`](incin_core::tensor::device::DeviceKind::Rocm) resolves
//! to a typed `Unsupported` answer. The HIP lane fills the table one kernel
//! at a time by adding entries to the `rocm_descriptor_operations!`
//! declaration; this module re-exports the single table so the declaration
//! stays the only source of group membership.

pub use crate::capability::ROCM_CAPABILITIES;

#[cfg(test)]
mod tests {
    use super::*;
    use incin_core::backend_authoring::{CapabilityQuery, OperationIdentity, SupportLevel};
    use incin_core::exec::{LayoutClass, MathMode};
    use incin_core::shapes::OperationKind;
    use incin_core::tensor::device::DeviceKind;
    use incin_core::tensor::dtype::DTypeId;

    fn query(operation: OperationKind) -> CapabilityQuery {
        CapabilityQuery {
            operation: OperationIdentity::Builtin(operation),
            dtype: DTypeId::F32.descriptor(),
            layout: LayoutClass::Contiguous,
            rank: 2,
            training: false,
            math_mode: MathMode::Precise,
        }
    }

    #[test]
    fn the_table_is_exactly_empty() {
        assert!(ROCM_CAPABILITIES.is_empty());
    }

    /// Every representative operation answers typed-unsupported, never native
    /// and never a panic. (Served by the registry's empty-table wildcard arm
    /// until the explicit `DeviceKind::Rocm` wiring lands; the assertion is
    /// phrased against the answer, not the arm, so it survives the wiring.)
    #[test]
    fn every_representative_operation_is_typed_unsupported() {
        for operation in [
            OperationKind::Storage,
            OperationKind::Pointwise,
            OperationKind::MatMul,
            OperationKind::Reduction,
            OperationKind::Broadcast,
            OperationKind::Reshape,
        ] {
            let level = crate::capability::support(DeviceKind::Rocm, &query(operation));
            assert!(
                matches!(level, SupportLevel::Unsupported(_)),
                "{operation:?} must be unsupported while no kernels exist"
            );
        }
    }

    /// The refusal carries a reason naming the operation, so a caller reading
    /// the answer knows what was asked, not just that it failed.
    #[test]
    fn the_unsupported_answer_names_its_operation() {
        let level = crate::capability::support(DeviceKind::Rocm, &query(OperationKind::MatMul));
        match level {
            SupportLevel::Unsupported(reason) => {
                assert!(
                    reason.to_string().to_lowercase().contains("matmul"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected Unsupported, got {other:?}"),
        }
    }
}
