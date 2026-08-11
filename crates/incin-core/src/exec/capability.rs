//! Backend-neutral capability queries and deterministic registry resolution.

use core::fmt;

use crate::exec::catalog::OperationKey;
use crate::exec::{LayoutClass, MathMode};
use crate::prelude::{DTypeDescriptor, OperationKind};

/// A complete runtime support question for one physical execution path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CapabilityQuery {
    pub operation: OperationKind,
    pub dtype: DTypeDescriptor,
    pub layout: LayoutClass,
    pub rank: usize,
    pub training: bool,
    pub math_mode: MathMode,
}

/// Capability query for an operation supplied outside the built-in catalog.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CustomCapabilityQuery {
    pub operation: OperationKey,
    pub dtype: DTypeDescriptor,
    pub layout: LayoutClass,
    pub rank: usize,
    pub training: bool,
    pub math_mode: MathMode,
}

/// How an advertised operation is implemented.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ImplementationKind {
    Native,
    Composed,
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnsupportedReason {
    Operation {
        operation: OperationKind,
    },
    CustomOperation {
        operation: OperationKey,
    },
    DType {
        operation: OperationKind,
        dtype: DTypeDescriptor,
    },
    Layout {
        operation: OperationKind,
        layout: LayoutClass,
    },
    Rank {
        operation: OperationKind,
        rank: usize,
        min: usize,
        max: usize,
    },
    Training {
        operation: OperationKind,
    },
    MathMode {
        operation: OperationKind,
        math_mode: MathMode,
    },
    MissingDeviceFeature {
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SupportLevel {
    Native,
    Composed,
    Fallback,
    Unsupported(UnsupportedReason),
}

impl SupportLevel {
    #[must_use]
    pub const fn is_supported(self) -> bool {
        matches!(self, Self::Native | Self::Composed | Self::Fallback)
    }

    #[must_use]
    pub const fn is_device_local(self) -> bool {
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
    Range { min: usize, max: usize },
}

/// One immutable capability registration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CapabilityRule {
    pub operation: OperationKind,
    pub dtypes: &'static [DTypeDescriptor],
    pub layouts: &'static [LayoutClass],
    pub min_rank: usize,
    pub max_rank: usize,
    pub training: bool,
    pub math_modes: &'static [MathMode],
    pub implementation: ImplementationKind,
}

impl CapabilityRule {
    #[must_use]
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
    pub const fn new(rules: &'static [CapabilityRule]) -> Self {
        Self { rules }
    }

    #[must_use]
    pub const fn registrations(self) -> &'static [CapabilityRule] {
        self.rules
    }

    fn operation_matches(rule: &CapabilityRule, query: &CapabilityQuery) -> bool {
        // Families classify work; they do not prove an exact implementation.
        // A backend must register the precise semantic identity it executes.
        rule.operation == query.operation
    }
}

/// Runtime capability inspection implemented by registries and later contexts.
pub trait Capabilities {
    fn support(&self, query: &CapabilityQuery) -> SupportLevel;

    /// Answers support for an open operation identity.
    ///
    /// The rejecting default preserves fail-closed behavior for backends that
    /// have not opted into a downstream operation.
    fn support_custom(&self, query: &CustomCapabilityQuery) -> SupportLevel {
        SupportLevel::Unsupported(UnsupportedReason::CustomOperation {
            operation: query.operation,
        })
    }
}

impl Capabilities for CapabilityRegistry {
    fn support(&self, query: &CapabilityQuery) -> SupportLevel {
        let mut operation = false;
        let mut dtype = false;
        let mut layout = false;
        let mut rank = false;
        let mut training = false;

        for rule in self.rules {
            if !Self::operation_matches(rule, query) {
                continue;
            }
            operation = true;
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

        let reason = if !operation {
            UnsupportedReason::Operation {
                operation: query.operation,
            }
        } else if !dtype {
            UnsupportedReason::DType {
                operation: query.operation,
                dtype: query.dtype,
            }
        } else if !layout {
            UnsupportedReason::Layout {
                operation: query.operation,
                layout: query.layout,
            }
        } else if !rank {
            let Some(rule) = self.rules.iter().find(|rule| {
                Self::operation_matches(rule, query)
                    && rule.dtypes.contains(&query.dtype)
                    && rule.layouts.contains(&query.layout)
            }) else {
                return SupportLevel::Unsupported(UnsupportedReason::Operation {
                    operation: query.operation,
                });
            };
            UnsupportedReason::Rank {
                operation: query.operation,
                rank: query.rank,
                min: rule.min_rank,
                max: rule.max_rank,
            }
        } else if !training {
            UnsupportedReason::Training {
                operation: query.operation,
            }
        } else {
            UnsupportedReason::MathMode {
                operation: query.operation,
                math_mode: query.math_mode,
            }
        };
        SupportLevel::Unsupported(reason)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::DTypeId;

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
            operation,
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
