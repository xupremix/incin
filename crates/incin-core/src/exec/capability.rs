//! Backend-neutral capability queries and deterministic registry resolution.

use alloc::borrow::ToOwned;
use core::fmt;

use crate::exec::catalog::OperationKey;
use crate::exec::{LayoutClass, MathMode};
use crate::shapes::error::OperationKind;
use crate::tensor::dtype::DTypeDescriptor;

/// The identity of one operation in the unified execution universe.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum OperationIdentity {
    /// Query a built-in catalog operation.
    Builtin(OperationKind),
    /// Query a custom-registered operation key.
    Custom(OperationKey),
}

impl OperationIdentity {
    /// Execution site this key runs at, when known.
    pub fn execution_site(&self) -> Option<super::catalog::ExecutionSite> {
        match self {
            Self::Builtin(operation) => {
                super::catalog::catalog_entry(*operation).map(|row| row.site)
            }
            Self::Custom(_) => None,
        }
    }

    /// Stable human-readable identity used by graph consumers.
    #[must_use]
    pub fn display_name(&self) -> alloc::string::String {
        match self {
            Self::Builtin(operation) => super::catalog::catalog_entry(*operation)
                .map(|entry| entry.name.to_owned())
                .unwrap_or_else(|| operation.name().to_owned()),
            Self::Custom(key) => alloc::format!("{}/{}@{}", key.namespace, key.name, key.version),
        }
    }

    /// Explicit ONNX projection for this identity.
    pub fn onnx_name(&self) -> Option<&'static str> {
        match self {
            Self::Builtin(operation) => super::catalog::onnx_name(*operation),
            Self::Custom(_) => None,
        }
    }
}

/// A complete runtime support question for one physical execution path.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CapabilityQuery {
    /// Operation identity the row describes.
    pub operation: OperationIdentity,
    /// Dtype the rule applies to.
    pub dtype: DTypeDescriptor,
    /// Layout class the rule accepts.
    pub layout: LayoutClass,
    /// Rank the rule accepts.
    pub rank: usize,
    /// Whether training-mode execution is covered.
    pub training: bool,
    /// Math mode the rule was registered under.
    pub math_mode: MathMode,
}

/// How an advertised operation is implemented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImplementationKind {
    /// Executed by the backend's own kernel.
    Native,
    /// Executed by composing other supported operations.
    Composed,
    /// Unsupported natively; only a fallback composition exists.
    Fallback,
}

impl ImplementationKind {
    /// The lowercase name used in generated documentation.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Native => "native",
            Self::Composed => "composed",
            Self::Fallback => "fallback",
        }
    }
}

/// Stable reason an otherwise valid tensor request cannot execute.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum UnsupportedReason {
    /// Query a built-in operation's support.
    Operation {
        /// Operation this query targets.
        operation: OperationKind,
    },
    /// Query a custom operation's support.
    CustomOperation {
        /// Custom operation this query targets.
        operation: OperationKey,
    },
    /// Ask whether an operation supports one dtype.
    DType {
        /// Operation this query targets.
        operation: OperationKind,
        /// Dtype the query asks about.
        dtype: DTypeDescriptor,
    },
    /// Ask whether an operation supports one layout class.
    Layout {
        /// Operation this query targets.
        operation: OperationKind,
        /// Layout class the query asks about.
        layout: LayoutClass,
    },
    /// Ask whether an operation supports one rank.
    Rank {
        /// Operation this query targets.
        operation: OperationKind,
        /// Rank the query asks about.
        rank: usize,
        /// Inclusive minimum accepted value.
        min: usize,
        /// Inclusive maximum accepted value.
        max: usize,
    },
    /// Ask whether an operation supports training execution.
    Training {
        /// Operation this query targets.
        operation: OperationKind,
    },
    /// Ask about math-mode support.
    MathMode {
        /// Operation this query targets.
        operation: OperationKind,
        /// Math mode asked about.
        math_mode: MathMode,
    },
    /// The backend lacks a compile-time feature this path needs.
    MissingDeviceFeature {
        /// Compile-time feature name that is absent.
        feature: &'static str,
    },
}

impl fmt::Display for UnsupportedReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Operation { operation } => write!(f, "operation {operation} is not registered"),
            Self::CustomOperation { operation } => write!(
                f,
                "custom operation {}/{} v{} is not registered",
                operation.namespace, operation.name, operation.version
            ),
            Self::DType { operation, dtype } => {
                write!(f, "dtype {} is unsupported for {operation}", dtype.name())
            }
            Self::Layout { operation, layout } => {
                write!(
                    f,
                    "layout {} is unsupported for {operation}",
                    layout.as_str()
                )
            }
            Self::Rank {
                operation,
                rank,
                min,
                max,
            } => {
                write!(
                    f,
                    "rank {rank} is unsupported for {operation}; expected {min}..={max}"
                )
            }
            Self::Training { operation } => {
                write!(f, "training is unsupported for {operation}")
            }
            Self::MathMode {
                operation,
                math_mode,
            } => {
                write!(
                    f,
                    "math mode {} is unsupported for {operation}",
                    math_mode.as_str()
                )
            }
            Self::MissingDeviceFeature { feature } => {
                write!(f, "required device feature {feature} is unavailable")
            }
        }
    }
}

/// Result of a capability query.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SupportLevel {
    /// Executed by the backend's own kernel.
    Native,
    /// Executed by composing other supported operations.
    Composed,
    /// Unsupported natively; only a fallback composition exists.
    Fallback,
    /// Unsupported; the reason states why.
    Unsupported(UnsupportedReason),
}

impl SupportLevel {
    #[must_use]
    /// True when native or composed support exists.
    pub fn is_supported(&self) -> bool {
        matches!(self, Self::Native | Self::Composed | Self::Fallback)
    }

    #[must_use]
    /// True when support does not depend on other devices.
    pub fn is_device_local(&self) -> bool {
        matches!(self, Self::Native | Self::Composed)
    }
}

impl From<ImplementationKind> for SupportLevel {
    fn from(kind: ImplementationKind) -> Self {
        match kind {
            ImplementationKind::Native => Self::Native,
            ImplementationKind::Composed => Self::Composed,
            ImplementationKind::Fallback => Self::Fallback,
        }
    }
}

/// Explicit backend rank support capability.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum RankSupport {
    /// Supports arbitrary ranks without a backend ceiling (e.g. CPU generic loop).
    Any,
    /// Supports ranks up to `max`.
    UpTo(usize),
    /// Supports ranks within `[min, max]`.
    Range {
        /// Inclusive minimum accepted value.
        min: usize,
        /// Inclusive maximum accepted value.
        max: usize,
    },
}

/// One immutable capability registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilityRule {
    /// Operation these rules describe.
    pub operation: OperationKind,
    /// Dtypes accepted for this operation.
    pub dtypes: &'static [DTypeDescriptor],
    /// Layout classes accepted for this operation.
    pub layouts: &'static [LayoutClass],
    /// Inclusive minimum supported rank.
    pub min_rank: usize,
    /// Inclusive maximum supported rank.
    pub max_rank: usize,
    /// Whether training-mode execution is covered.
    pub training: bool,
    /// Math modes accepted for this operation.
    pub math_modes: &'static [MathMode],
    /// How the backend realizes the operation.
    pub implementation: ImplementationKind,
}

impl CapabilityRule {
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    /// Creates a rule from every acceptance field.
    pub const fn new(
        operation: OperationKind,
        dtypes: &'static [DTypeDescriptor],
        layouts: &'static [LayoutClass],
        min_rank: usize,
        max_rank: usize,
        training: bool,
        math_modes: &'static [MathMode],
        implementation: ImplementationKind,
    ) -> Self {
        Self {
            operation,
            dtypes,
            layouts,
            min_rank,
            max_rank,
            training,
            math_modes,
            implementation,
        }
    }

    /// Returns the explicit rank support for this capability rule.
    #[must_use]
    pub const fn rank_support(&self) -> RankSupport {
        if self.min_rank == 0 && self.max_rank == usize::MAX {
            RankSupport::Any
        } else if self.min_rank == 0 {
            RankSupport::UpTo(self.max_rank)
        } else {
            RankSupport::Range {
                min: self.min_rank,
                max: self.max_rank,
            }
        }
    }
}

/// A queryable, allocation-free view over versioned backend registrations.
#[derive(Debug, Clone, Copy)]
pub struct CapabilityRegistry {
    rules: &'static [CapabilityRule],
}

impl CapabilityRegistry {
    #[must_use]
    /// Builds a registry from its static rule table.
    pub const fn new(rules: &'static [CapabilityRule]) -> Self {
        Self { rules }
    }

    #[must_use]
    /// Borrows the static rule table backing this registry.
    pub const fn registrations(self) -> &'static [CapabilityRule] {
        self.rules
    }

    fn operation_matches(rule: &CapabilityRule, query: &CapabilityQuery) -> bool {
        // Families classify work; they do not prove an exact implementation.
        // A backend must register the precise semantic identity it executes.
        matches!(query.operation, OperationIdentity::Builtin(operation) if rule.operation == operation)
    }
}

/// Runtime capability inspection implemented by registries and later contexts.
pub trait Capabilities {
    /// Answer a capability query against this registry's rules.
    fn support(&self, query: &CapabilityQuery) -> SupportLevel;
}

impl Capabilities for CapabilityRegistry {
    fn support(&self, query: &CapabilityQuery) -> SupportLevel {
        let OperationIdentity::Builtin(operation) = &query.operation else {
            let OperationIdentity::Custom(operation) = &query.operation else {
                unreachable!("operation identity matched above")
            };
            return SupportLevel::Unsupported(UnsupportedReason::CustomOperation {
                operation: operation.clone(),
            });
        };
        let mut operation_found = false;
        let mut dtype = false;
        let mut layout = false;
        let mut rank = false;
        let mut training = false;

        for rule in self.rules {
            if !Self::operation_matches(rule, query) {
                continue;
            }
            operation_found = true;
            if !rule.dtypes.contains(&query.dtype) {
                continue;
            }
            dtype = true;
            if !rule.layouts.contains(&query.layout) {
                continue;
            }
            layout = true;
            if !(rule.min_rank..=rule.max_rank).contains(&query.rank) {
                continue;
            }
            rank = true;
            if query.training && !rule.training {
                continue;
            }
            training = true;
            if !rule.math_modes.contains(&query.math_mode) {
                continue;
            }
            return rule.implementation.into();
        }

        let reason = if !operation_found {
            UnsupportedReason::Operation {
                operation: *operation,
            }
        } else if !dtype {
            UnsupportedReason::DType {
                operation: *operation,
                dtype: query.dtype,
            }
        } else if !layout {
            UnsupportedReason::Layout {
                operation: *operation,
                layout: query.layout,
            }
        } else if !rank {
            let Some(rule) = self.rules.iter().find(|rule| {
                Self::operation_matches(rule, query)
                    && rule.dtypes.contains(&query.dtype)
                    && rule.layouts.contains(&query.layout)
            }) else {
                return SupportLevel::Unsupported(UnsupportedReason::Operation {
                    operation: *operation,
                });
            };
            UnsupportedReason::Rank {
                operation: *operation,
                rank: query.rank,
                min: rule.min_rank,
                max: rule.max_rank,
            }
        } else if !training {
            UnsupportedReason::Training {
                operation: *operation,
            }
        } else {
            UnsupportedReason::MathMode {
                operation: *operation,
                math_mode: query.math_mode,
            }
        };
        SupportLevel::Unsupported(reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::dtype::DTypeId;

    const F32_DESC: DTypeDescriptor = DTypeId::F32.descriptor();
    const F32: &[DTypeDescriptor] = &[F32_DESC];
    const CONTIGUOUS: &[LayoutClass] = &[LayoutClass::Contiguous];
    const PRECISE: &[MathMode] = &[MathMode::Precise];
    const RULES: &[CapabilityRule] = &[
        CapabilityRule::new(
            OperationKind::Reduction,
            F32,
            CONTIGUOUS,
            1,
            8,
            true,
            PRECISE,
            ImplementationKind::Native,
        ),
        CapabilityRule::new(
            OperationKind::MatMul,
            F32,
            CONTIGUOUS,
            2,
            4,
            false,
            PRECISE,
            ImplementationKind::Composed,
        ),
    ];

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
    fn families_never_imply_exact_support() {
        let registry = CapabilityRegistry::new(RULES);
        assert_eq!(
            registry.support(&query(OperationKind::Reduction)),
            SupportLevel::Native
        );
        assert_eq!(
            registry.support(&query(OperationKind::MatMul)),
            SupportLevel::Composed
        );
        let mut training = query(OperationKind::MatMul);
        training.training = true;
        assert!(matches!(
            registry.support(&training),
            SupportLevel::Unsupported(UnsupportedReason::Training { .. })
        ));
        assert!(matches!(
            registry.support(&query(OperationKind::SumDim)),
            SupportLevel::Unsupported(UnsupportedReason::Operation {
                operation: OperationKind::SumDim
            })
        ));
    }

    #[test]
    fn rejection_identifies_the_first_unsatisfied_constraint() {
        let registry = CapabilityRegistry::new(RULES);
        let mut q = query(OperationKind::Reduction);
        q.dtype = DTypeId::F64.descriptor();
        let level = registry.support(&q);
        if let SupportLevel::Unsupported(UnsupportedReason::DType { dtype, .. }) = level {
            assert_eq!(dtype, DTypeId::F64.descriptor());
        } else {
            panic!("expected DType unsupported, got {:?}", level);
        }
        q.dtype = DTypeId::F32.descriptor();
        q.layout = LayoutClass::Strided;
        assert!(matches!(
            registry.support(&q),
            SupportLevel::Unsupported(UnsupportedReason::Layout {
                layout: LayoutClass::Strided,
                ..
            })
        ));
    }
}
