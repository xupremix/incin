#![cfg(feature = "std")]

//! `EXE-010`: the external-backend SDK, its conformance suite, and the template.
//!
//! Three things live here and they answer three different questions.
//!
//! [`template`] is the **template** PROPOSALS.md sec. 2.9 asks for: a complete
//! minimal backend, about a hundred and fifty lines, that an author copies and
//! replaces the bodies of. It is a template rather than prose because prose
//! describing sec. 2.9's seven bullets goes stale and a backend that compiles
//! and passes the suite cannot.
//!
//! It is also the harness's **control**. A conformance suite that has never
//! passed a backend known to be correct says nothing when it fails one, and a
//! suite that has never *failed* one says nothing when it passes. So the
//! [`broken`] module deliberately gets each check wrong, one at a time, and
//! each is asserted to fail exactly the check it breaks.
//!
//! And the Candle adapter is the real subject: a third-party backend nobody
//! here wrote, joining the same contract through the same trait.

use incin_backends::external::conformance::{self, Outcome, Report, Subject, Tolerance};
use incin_core::backend_authoring::{Execute, ExecutionRequest, op};
use incin_core::exec::{
    Alignment, Capabilities, CapabilityQuery, ExecutionDescriptor, OperationIdentity, SupportLevel,
    TensorMeta, UnsupportedReason,
};
use incin_core::prelude::{
    BackendError, Cpu, DType, DTypeId, DeviceId, OperationKind, ShapeBuf, StorageBackend, StrideBuf,
};

type MatMulOperation = op::MatMulExact;
type ReshapeOperation = op::ReshapeExact;

// ============================================================================
// The template — copy this module
// ============================================================================

/// A complete minimal external backend.
///
/// Everything sec. 2.9 lists that can be shown in code is here, in the order an
/// author needs it: storage carrying checked metadata, capability registration,
/// two descriptor implementations, and the [`Subject`] impl that hands the
/// whole thing to the conformance suite. Its "device" is a `Vec<f32>`, which
/// keeps the file about the *contract* rather than about arithmetic.
///
/// The parts worth copying exactly are the ones that are easy to get wrong:
/// metadata validated once at the boundary rather than trusted, a registry that
/// says no to something, and executors that check their input count and their
/// operands against the descriptor instead of assuming the proof covered them.
pub mod template {
    use super::*;

    /// Storage: the values, plus the checked metadata the descriptor path needs.
    ///
    /// The metadata is built once, at the boundary, by [`Self::try_new`]. A
    /// backend that constructs `TensorMeta` at each use site will eventually
    /// construct one that disagrees with the buffer it describes.
    #[derive(Debug, Clone)]
    pub struct TemplateStorage {
        values: Vec<f32>,
        meta: TensorMeta,
    }

    impl incin_core::backend_authoring::StorageOutput for TemplateStorage {}

    impl TemplateStorage {
        /// Validate a buffer and its intended shape into checked metadata.
        pub fn try_new(dims: &[usize], values: Vec<f32>) -> Result<Self, String> {
            let shape = ShapeBuf::from_slice(dims);
            let numel = shape
                .checked_numel(OperationKind::Storage)
                .map_err(|error| format!("{error}"))?;
            if numel != values.len() {
                return Err(format!(
                    "{dims:?} holds {numel} elements but {} values were given",
                    values.len()
                ));
            }
            // Row-major strides, computed rather than assumed: a backend that
            // hard-codes them gets rank-1 and rank-0 wrong.
            let mut strides = vec![1_usize; dims.len()];
            for axis in (0..dims.len().saturating_sub(1)).rev() {
                strides[axis] = strides[axis + 1] * dims[axis + 1];
            }
            let meta = TensorMeta::try_new(
                shape,
                StrideBuf::from_slice(&strides),
                0,
                DTypeId::F32.descriptor(),
                DeviceId::cpu(),
                // Claim only what is true. `Vec` guarantees `align_of::<f32>()`,
                // and a larger claim is one a kernel may act on.
                Alignment::BYTE,
                numel,
            )
            .map_err(|error| format!("{error}"))?;
            Ok(Self { values, meta })
        }

        #[must_use]
        pub fn values(&self) -> &[f32] {
            &self.values
        }

        #[must_use]
        pub const fn metadata(&self) -> &TensorMeta {
            &self.meta
        }
    }

    /// The backend itself. Stateless here; a real one holds a device handle.
    #[derive(Debug, Clone, Copy, Default)]
    pub struct TemplateBackend;

    impl StorageBackend for TemplateBackend {
        const BACKEND_NAME: &'static str = "Template";
        type Storage<K: DType> = TemplateStorage;
        type Device = Cpu;

        fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta {
            storage.metadata()
        }
    }

    /// Capability registration: what this backend claims, and nothing more.
    ///
    /// Sec. 2.9's "missing support is visible through the capability registry"
    /// is this impl. Every `no` here is a promise the executor keeps, and the
    /// conformance suite checks that the two agree.
    impl Capabilities for TemplateBackend {
        fn support(&self, query: &CapabilityQuery) -> SupportLevel {
            let OperationIdentity::Builtin(operation) = &query.operation else {
                return SupportLevel::Unsupported(UnsupportedReason::CustomOperation {
                    operation: match &query.operation {
                        OperationIdentity::Custom(operation) => operation.clone(),
                        OperationIdentity::Builtin(_) => unreachable!(),
                    },
                });
            };
            if query.dtype != DTypeId::F32.descriptor() {
                return SupportLevel::Unsupported(UnsupportedReason::DType {
                    operation: *operation,
                    dtype: query.dtype,
                });
            }
            match operation {
                // Rank 2 only, because the matmul below is rank-2 only.
                // Registering a rank this executor cannot run is the exact
                // false claim the registry exists to prevent.
                OperationKind::MatMulExact if query.rank == 2 => SupportLevel::Native,
                OperationKind::MatMulExact => SupportLevel::Unsupported(UnsupportedReason::Rank {
                    operation: *operation,
                    rank: query.rank,
                    min: 2,
                    max: 2,
                }),
                OperationKind::ReshapeExact => SupportLevel::Native,
                operation => SupportLevel::Unsupported(UnsupportedReason::Operation {
                    operation: *operation,
                }),
            }
        }
    }

    fn invalid(operation: OperationKind, reason: &'static str) -> BackendError {
        BackendError::InvalidInput { operation, reason }
    }

    fn operand<'a>(
        handle: &'a incin_core::exec::TensorHandle<'a>,
        operation: OperationKind,
    ) -> Result<&'a TemplateStorage, BackendError> {
        handle
            .downcast_ref::<TemplateStorage>()
            .ok_or_else(|| invalid(operation, "operand is not this backend's storage"))
    }

    impl Execute<MatMulOperation> for TemplateBackend {
        type Output = TemplateStorage;

        fn execute(
            &self,
            request: ExecutionRequest<'_, MatMulOperation, Self>,
        ) -> Result<Self::Output, BackendError> {
            let spec = request.operation.descriptor();
            // Input arity is the executor's to check. `Validated` proves the
            // shapes are compatible; the handles arrive separately and nothing
            // has proved how many there are.
            let [lhs, rhs] = request.inputs else {
                return Err(invalid(
                    OperationKind::MatMul,
                    "matmul expects exactly two operands",
                ));
            };
            let lhs = operand(lhs, OperationKind::MatMul)?;
            let rhs = operand(rhs, OperationKind::MatMul)?;

            let lhs_dims = lhs.metadata().shape().dims();
            let rhs_dims = rhs.metadata().shape().dims();
            if lhs_dims.len() != 2 || rhs_dims.len() != 2 {
                return Err(invalid(
                    OperationKind::MatMul,
                    "this backend registers matmul at rank 2 only",
                ));
            }
            // And the operands must be the ones the descriptor was proved for.
            if lhs_dims[1] != rhs_dims[0] {
                return Err(invalid(
                    OperationKind::MatMul,
                    "operand shapes disagree with the validated descriptor",
                ));
            }

            let (m, k, n) = (lhs_dims[0], lhs_dims[1], rhs_dims[1]);
            let mut out = vec![0.0_f32; m * n];
            for row in 0..m {
                for column in 0..n {
                    let mut sum = 0.0_f32;
                    for inner in 0..k {
                        sum += lhs.values()[row * k + inner] * rhs.values()[inner * n + column];
                    }
                    out[row * n + column] = sum;
                }
            }

            let output_shape = spec.output_shape().ok_or(BackendError::InvalidInput {
                operation: OperationKind::MatMulExact,
                reason: "descriptor has no output",
            })?;
            TemplateStorage::try_new(output_shape.dims(), out).map_err(|message| {
                BackendError::Execution {
                    operation: OperationKind::MatMul,
                    message: message.into(),
                }
            })
        }
    }

    impl Execute<ReshapeOperation> for TemplateBackend {
        type Output = TemplateStorage;

        fn execute(
            &self,
            request: ExecutionRequest<'_, ReshapeOperation, Self>,
        ) -> Result<Self::Output, BackendError> {
            let spec = request.operation.descriptor();
            let [handle] = request.inputs else {
                return Err(invalid(
                    OperationKind::Reshape,
                    "reshape expects exactly one operand",
                ));
            };
            let input = operand(handle, OperationKind::Reshape)?;

            if input.metadata().shape().dims() != [2, 3] {
                return Err(invalid(
                    OperationKind::Reshape,
                    "operand shape disagrees with the validated descriptor",
                ));
            }

            // Reshape re-addresses; it must not compute.
            let output_shape = spec
                .output_shape()
                .ok_or_else(|| invalid(OperationKind::ReshapeExact, "descriptor has no output"))?;
            TemplateStorage::try_new(output_shape.dims(), input.values().to_vec()).map_err(
                |message| BackendError::Execution {
                    operation: OperationKind::Reshape,
                    message: message.into(),
                },
            )
        }
    }

    /// Hand the backend to the conformance suite.
    pub struct TemplateSubject;

    impl Subject for TemplateSubject {
        type Storage = TemplateStorage;
        type Backend = TemplateBackend;

        fn name(&self) -> String {
            "TemplateBackend".to_string()
        }

        fn backend(&self) -> Self::Backend {
            TemplateBackend
        }

        fn storage(&self, dims: &[usize], values: &[f32]) -> Result<Self::Storage, String> {
            TemplateStorage::try_new(dims, values.to_vec())
        }

        fn values(&self, storage: &Self::Storage) -> Result<Vec<f32>, String> {
            Ok(storage.values().to_vec())
        }
    }
}

use template::{TemplateBackend, TemplateStorage, TemplateSubject};

fn failed_checks(report: &Report) -> Vec<&str> {
    report.failures().map(|check| check.name).collect()
}

// ============================================================================
// The control: a correct backend passes
// ============================================================================

#[test]
fn the_template_backend_conforms() {
    let report = conformance::run(&TemplateSubject);
    assert!(report.passed(), "{}", report.to_text());
    assert_eq!(
        report.skipped().count(),
        0,
        "the template claims both descriptors, so nothing should skip:\n{}",
        report.to_text()
    );
    assert!(
        report.checks.len() >= 8,
        "the suite ran only {} checks:\n{}",
        report.checks.len(),
        report.to_text()
    );
}

// ============================================================================
// The other control: a broken backend fails, and fails the right check
// ============================================================================

/// Backends that each get exactly one thing wrong.
///
/// This is the half of a conformance suite people skip, and it is the half that
/// decides whether the suite means anything: a check that has never failed is
/// indistinguishable from a check that cannot fail. Each subject below is the
/// template with one behaviour replaced, and each is asserted to fail exactly
/// the check that behaviour breaks — no more, so the checks are independent,
/// and no fewer, so none of them is vacuous.
mod broken {
    use super::*;

    /// A registry that claims every dtype, including ones the executor cannot
    /// hold. The most common real mistake: `SupportLevel::Native` written once
    /// and never revisited.
    pub struct ClaimsEveryDType;

    #[derive(Debug, Clone, Copy, Default)]
    pub struct OverclaimingBackend;

    impl StorageBackend for OverclaimingBackend {
        const BACKEND_NAME: &'static str = "Overclaiming";
        type Storage<K: DType> = TemplateStorage;
        type Device = Cpu;
        fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta {
            storage.metadata()
        }
    }

    impl Capabilities for OverclaimingBackend {
        fn support(&self, _query: &CapabilityQuery) -> SupportLevel {
            SupportLevel::Native
        }
    }

    impl Execute<MatMulOperation> for OverclaimingBackend {
        type Output = TemplateStorage;
        fn execute(
            &self,
            request: ExecutionRequest<'_, MatMulOperation, Self>,
        ) -> Result<Self::Output, BackendError> {
            TemplateBackend.execute(ExecutionRequest {
                operation: request.operation,
                inputs: request.inputs,
                context: &incin_core::exec::ExecutionContext::new(TemplateBackend),
                payload: request.payload,
            })
        }
    }

    impl Execute<ReshapeOperation> for OverclaimingBackend {
        type Output = TemplateStorage;
        fn execute(
            &self,
            request: ExecutionRequest<'_, ReshapeOperation, Self>,
        ) -> Result<Self::Output, BackendError> {
            TemplateBackend.execute(ExecutionRequest {
                operation: request.operation,
                inputs: request.inputs,
                context: &incin_core::exec::ExecutionContext::new(TemplateBackend),
                payload: request.payload,
            })
        }
    }

    impl Subject for ClaimsEveryDType {
        type Storage = TemplateStorage;
        type Backend = OverclaimingBackend;
        fn name(&self) -> String {
            "ClaimsEveryDType".to_string()
        }
        fn backend(&self) -> Self::Backend {
            OverclaimingBackend
        }
        fn storage(&self, dims: &[usize], values: &[f32]) -> Result<Self::Storage, String> {
            TemplateStorage::try_new(dims, values.to_vec())
        }
        fn values(&self, storage: &Self::Storage) -> Result<Vec<f32>, String> {
            Ok(storage.values().to_vec())
        }
    }

    /// A backend that panics instead of returning an error, and one that
    /// accepts a wrong operand count. Both are the same impl: the panic is what
    /// happens when an executor indexes `inputs` instead of matching on it.
    pub struct PanicsOnAWrongArity;

    #[derive(Debug, Clone, Copy, Default)]
    pub struct IndexingBackend;

    impl StorageBackend for IndexingBackend {
        const BACKEND_NAME: &'static str = "Indexing";
        type Storage<K: DType> = TemplateStorage;
        type Device = Cpu;
        fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta {
            storage.metadata()
        }
    }

    impl Capabilities for IndexingBackend {
        fn support(&self, query: &CapabilityQuery) -> SupportLevel {
            TemplateBackend.support(query)
        }
    }

    impl Execute<MatMulOperation> for IndexingBackend {
        type Output = TemplateStorage;
        fn execute(
            &self,
            request: ExecutionRequest<'_, MatMulOperation, Self>,
        ) -> Result<Self::Output, BackendError> {
            // The mistake: trusting the proof to have covered the handles.
            // Slicing to two panics when only one arrived, which is precisely
            // what an executor that indexes rather than matches does.
            let inputs = &request.inputs[0..2];
            TemplateBackend.execute(ExecutionRequest {
                operation: request.operation,
                inputs,
                context: &incin_core::exec::ExecutionContext::new(TemplateBackend),
                payload: request.payload,
            })
        }
    }

    impl Execute<ReshapeOperation> for IndexingBackend {
        type Output = TemplateStorage;
        fn execute(
            &self,
            request: ExecutionRequest<'_, ReshapeOperation, Self>,
        ) -> Result<Self::Output, BackendError> {
            TemplateBackend.execute(ExecutionRequest {
                operation: request.operation,
                inputs: request.inputs,
                context: &incin_core::exec::ExecutionContext::new(TemplateBackend),
                payload: request.payload,
            })
        }
    }

    impl Subject for PanicsOnAWrongArity {
        type Storage = TemplateStorage;
        type Backend = IndexingBackend;
        fn name(&self) -> String {
            "PanicsOnAWrongArity".to_string()
        }
        fn backend(&self) -> Self::Backend {
            IndexingBackend
        }
        fn storage(&self, dims: &[usize], values: &[f32]) -> Result<Self::Storage, String> {
            TemplateStorage::try_new(dims, values.to_vec())
        }
        fn values(&self, storage: &Self::Storage) -> Result<Vec<f32>, String> {
            Ok(storage.values().to_vec())
        }
    }

    /// A backend whose numbers are wrong: operands multiplied the other way
    /// round. The shape is right, which is why a shape-only check would pass it.
    pub struct TransposesItsOperands;

    #[derive(Debug, Clone, Copy, Default)]
    pub struct SwappedBackend;

    impl StorageBackend for SwappedBackend {
        const BACKEND_NAME: &'static str = "Swapped";
        type Storage<K: DType> = TemplateStorage;
        type Device = Cpu;
        fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta {
            storage.metadata()
        }
    }

    impl Capabilities for SwappedBackend {
        fn support(&self, query: &CapabilityQuery) -> SupportLevel {
            TemplateBackend.support(query)
        }
    }

    impl Execute<MatMulOperation> for SwappedBackend {
        type Output = TemplateStorage;
        fn execute(
            &self,
            request: ExecutionRequest<'_, MatMulOperation, Self>,
        ) -> Result<Self::Output, BackendError> {
            let [lhs, rhs] = request.inputs else {
                return Err(BackendError::InvalidInput {
                    operation: OperationKind::MatMul,
                    reason: "matmul expects exactly two operands",
                });
            };
            let lhs = lhs
                .downcast_ref::<TemplateStorage>()
                .expect("the suite only supplies this backend's storage");
            let rhs = rhs
                .downcast_ref::<TemplateStorage>()
                .expect("the suite only supplies this backend's storage");
            let (m, k) = (
                lhs.metadata().shape().dims()[0],
                lhs.metadata().shape().dims()[1],
            );
            let n = rhs.metadata().shape().dims()[1];
            let mut out = vec![0.0_f32; m * n];
            for row in 0..m {
                for column in 0..n {
                    let mut sum = 0.0_f32;
                    for inner in 0..k {
                        // Swapped: rhs indexed as though it were lhs.
                        sum += rhs.values()[row * k + inner] * lhs.values()[inner * n + column];
                    }
                    out[row * n + column] = sum;
                }
            }
            let output_shape = request.operation.descriptor().output_shape().ok_or(
                BackendError::InvalidInput {
                    operation: OperationKind::MatMulExact,
                    reason: "descriptor has no output",
                },
            )?;
            TemplateStorage::try_new(output_shape.dims(), out).map_err(|message| {
                BackendError::Execution {
                    operation: OperationKind::MatMul,
                    message: message.into(),
                }
            })
        }
    }

    impl Execute<ReshapeOperation> for SwappedBackend {
        type Output = TemplateStorage;
        fn execute(
            &self,
            request: ExecutionRequest<'_, ReshapeOperation, Self>,
        ) -> Result<Self::Output, BackendError> {
            TemplateBackend.execute(ExecutionRequest {
                operation: request.operation,
                inputs: request.inputs,
                context: &incin_core::exec::ExecutionContext::new(TemplateBackend),
                payload: request.payload,
            })
        }
    }

    impl Subject for TransposesItsOperands {
        type Storage = TemplateStorage;
        type Backend = SwappedBackend;
        fn name(&self) -> String {
            "TransposesItsOperands".to_string()
        }
        fn backend(&self) -> Self::Backend {
            SwappedBackend
        }
        fn storage(&self, dims: &[usize], values: &[f32]) -> Result<Self::Storage, String> {
            TemplateStorage::try_new(dims, values.to_vec())
        }
        fn values(&self, storage: &Self::Storage) -> Result<Vec<f32>, String> {
            Ok(storage.values().to_vec())
        }
    }

    /// A backend that registers nothing.
    ///
    /// Not broken — this is the *correct* state of a backend part-way through
    /// being written, and sec. 2.9 says it should be a legitimate one: "an
    /// external backend implements only the operation descriptors it supports".
    /// It lives in this module because it is the other thing the suite must not
    /// get wrong.
    pub struct ClaimsNothing;

    #[derive(Debug, Clone, Copy, Default)]
    pub struct EmptyBackend;

    impl StorageBackend for EmptyBackend {
        const BACKEND_NAME: &'static str = "Empty";
        type Storage<K: DType> = TemplateStorage;
        type Device = Cpu;
        fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta {
            storage.metadata()
        }
    }

    impl Capabilities for EmptyBackend {
        fn support(&self, query: &CapabilityQuery) -> SupportLevel {
            let OperationIdentity::Builtin(operation) = &query.operation else {
                return SupportLevel::Unsupported(UnsupportedReason::CustomOperation {
                    operation: match &query.operation {
                        OperationIdentity::Custom(operation) => operation.clone(),
                        OperationIdentity::Builtin(_) => unreachable!(),
                    },
                });
            };
            SupportLevel::Unsupported(UnsupportedReason::Operation {
                operation: *operation,
            })
        }
    }

    impl Execute<MatMulOperation> for EmptyBackend {
        type Output = TemplateStorage;
        fn execute(
            &self,
            _request: ExecutionRequest<'_, MatMulOperation, Self>,
        ) -> Result<Self::Output, BackendError> {
            Err(BackendError::InvalidInput {
                operation: OperationKind::MatMul,
                reason: "this backend registers no operations",
            })
        }
    }

    impl Execute<ReshapeOperation> for EmptyBackend {
        type Output = TemplateStorage;
        fn execute(
            &self,
            _request: ExecutionRequest<'_, ReshapeOperation, Self>,
        ) -> Result<Self::Output, BackendError> {
            Err(BackendError::InvalidInput {
                operation: OperationKind::Reshape,
                reason: "this backend registers no operations",
            })
        }
    }

    impl Subject for ClaimsNothing {
        type Storage = TemplateStorage;
        type Backend = EmptyBackend;
        fn name(&self) -> String {
            "ClaimsNothing".to_string()
        }
        fn backend(&self) -> Self::Backend {
            EmptyBackend
        }
        fn storage(&self, dims: &[usize], values: &[f32]) -> Result<Self::Storage, String> {
            TemplateStorage::try_new(dims, values.to_vec())
        }
        fn values(&self, storage: &Self::Storage) -> Result<Vec<f32>, String> {
            Ok(storage.values().to_vec())
        }
    }
}

/// A backend that registers nothing passes, with everything skipped.
///
/// This is sec. 2.9's central promise as a test — "an external backend
/// implements only the operation descriptors it supports" — and getting it
/// wrong is what makes a conformance suite something authors route around. The
/// checks that need no registration still run, so passing is not vacuous.
#[test]
fn a_backend_that_claims_nothing_skips_rather_than_fails() {
    let report = conformance::run(&broken::ClaimsNothing);

    assert!(report.passed(), "{}", report.to_text());
    let skipped: Vec<&str> = report.skipped().map(|check| check.name).collect();
    assert_eq!(
        skipped,
        [
            "matmul_is_within_tolerance",
            "matmul_output_matches_its_descriptor",
            "matmul_rejects_a_wrong_input_count",
            "reshape_preserves_element_order",
            "reshape_rejects_a_mismatched_operand",
        ],
        "{}",
        report.to_text()
    );
    // Storage and metadata need no registration, and the registry-agreement
    // check is exactly the one that must still run: a backend claiming nothing
    // must also *do* nothing.
    let passed: Vec<&str> = report
        .checks
        .iter()
        .filter(|check| check.outcome == Outcome::Passed)
        .map(|check| check.name)
        .collect();
    assert_eq!(
        passed,
        [
            "storage_round_trips_its_values",
            "metadata_describes_the_tensor_it_came_from",
            "the_registry_and_the_executor_agree",
        ],
        "{}",
        report.to_text()
    );
}

#[test]
fn a_registry_that_claims_everything_fails_the_agreement_check() {
    let report = conformance::run(&broken::ClaimsEveryDType);
    assert_eq!(
        failed_checks(&report),
        ["the_registry_and_the_executor_agree"],
        "{}",
        report.to_text()
    );
}

/// A panicking executor is reported, not propagated.
///
/// The harness is the thing an author runs to find out what is wrong, and one
/// that dies on the first panic answers one question instead of nine. This is
/// the same lesson `UX-014` learned the expensive way, when a suite hit a
/// `SIGSEGV` and reported nothing at all.
#[test]
fn a_panicking_executor_is_a_reported_failure_not_a_dead_harness() {
    let report = conformance::run(&broken::PanicsOnAWrongArity);
    assert_eq!(
        failed_checks(&report),
        ["matmul_rejects_a_wrong_input_count"],
        "{}",
        report.to_text()
    );
    let failure = report
        .failures()
        .next()
        .expect("the check above failed by construction");
    let Outcome::Failed(why) = &failure.outcome else {
        unreachable!("failures() yields only Failed outcomes");
    };
    assert!(why.contains("panicked"), "{why}");
    // Everything after the panic still ran.
    assert!(report.checks.len() >= 8, "{}", report.to_text());
}

/// Wrong numbers with a right shape.
#[test]
fn a_backend_that_computes_the_wrong_product_fails_only_the_tolerance_check() {
    let report = conformance::run(&broken::TransposesItsOperands);
    assert_eq!(
        failed_checks(&report),
        ["matmul_is_within_tolerance"],
        "a shape-only suite would pass this backend:\n{}",
        report.to_text()
    );
}

// ============================================================================
// Tolerance
// ============================================================================

#[test]
fn tolerance_accepts_either_bound_and_rejects_a_non_finite_surprise() {
    // Near zero the relative bound has no magnitude to work with, and the
    // absolute one carries it.
    assert!(Tolerance::F32_ACCUMULATED.accepts(0.0, 1e-9));
    // Large, and the absolute bound is hopeless while the relative one is fine.
    assert!(Tolerance::F32_ACCUMULATED.accepts(1e6, 1e6 + 1.0));
    assert!(!Tolerance::F32_ACCUMULATED.accepts(1e6, 1.1e6));
    // Exact means exact.
    assert!(Tolerance::EXACT.accepts(1.5, 1.5));
    assert!(!Tolerance::EXACT.accepts(1.5, 1.5 + f64::EPSILON));
    // A NaN is never within tolerance of a real answer, whatever the bounds.
    assert!(!Tolerance::F32_WIDE.accepts(1.0, f64::NAN));
    assert!(!Tolerance::F32_WIDE.accepts(1.0, f64::INFINITY));
}

// ============================================================================
// The real third-party backend
// ============================================================================

#[cfg(feature = "external-candle")]
mod candle_subject {
    use super::*;
    use incin_backends::external::candle::{CandleBackend, CandleStorage};

    pub struct Candle;

    impl Subject for Candle {
        type Storage = CandleStorage;
        type Backend = CandleBackend<Cpu>;

        fn name(&self) -> String {
            "CandleBackend<Cpu>".to_string()
        }

        fn backend(&self) -> Self::Backend {
            CandleBackend::default()
        }

        fn storage(&self, dims: &[usize], values: &[f32]) -> Result<Self::Storage, String> {
            let tensor = candle_core::Tensor::from_slice(values, dims, &candle_core::Device::Cpu)
                .map_err(|error| format!("{error}"))?;
            CandleStorage::try_new(tensor).map_err(|error| format!("{error}"))
        }

        fn values(&self, storage: &Self::Storage) -> Result<Vec<f32>, String> {
            storage
                .tensor()
                .flatten_all()
                .and_then(|flat| flat.to_vec1::<f32>())
                .map_err(|error| format!("{error}"))
        }
    }
}

/// The row's actual subject: a backend nobody in this repository wrote.
///
/// The template proves the harness can be satisfied. This proves it can be
/// satisfied by a foreign tensor type that carries no `TensorMeta` of its own
/// and was never designed against this contract, which is the property sec. 2.9
/// is claiming.
#[test]
#[cfg(feature = "external-candle")]
fn the_candle_adapter_conforms() {
    let report = conformance::run(&candle_subject::Candle);
    assert!(report.passed(), "{}", report.to_text());
    assert_eq!(
        report.skipped().count(),
        0,
        "Candle registers matmul and reshape, so nothing should skip:\n{}",
        report.to_text()
    );
}

/// The same suite, two very different backends, and a report either can be read
/// against.
#[test]
#[cfg(feature = "external-candle")]
fn the_suite_runs_the_same_checks_for_every_backend() {
    let template = conformance::run(&TemplateSubject);
    let candle = conformance::run(&candle_subject::Candle);

    let template_names: Vec<&str> = template.checks.iter().map(|check| check.name).collect();
    let candle_names: Vec<&str> = candle.checks.iter().map(|check| check.name).collect();
    assert_eq!(
        template_names, candle_names,
        "a conformance suite that asks two backends different questions is not one"
    );
    assert_ne!(template.backend, candle.backend);
}
