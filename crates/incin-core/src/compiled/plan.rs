//! Immutable compiled execution plans and dynamic runtime guards.

use alloc::string::String;
use alloc::vec::Vec;

use crate::compiled::capture::CapturedGraph;
use crate::exec::{ShapeExpr, SymbolEnvironment, SymbolTable};
use crate::graph::ValueId;
use crate::prelude::{DTypeDescriptor, DTypeId, Error, Result};

/// Operation fusion strategy for graph compilation.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum FusionPolicy {
    /// Kernel fusion enabled (default).
    #[default]
    Enabled,
    /// Kernel fusion disabled for debugging or exact execution tracing.
    Disabled,
}

/// Dynamic shape handling policy.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum DynamicShapePolicy {
    /// Runtime shape and dtype verification using guard checks (default).
    #[default]
    Guarded,
    /// Strict static shape validation matching compiled dimensions exactly.
    Strict,
}

/// Options controlling graph compilation.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub struct CompileOptions {
    /// Kernel fusion policy.
    pub fusion: FusionPolicy,
    /// Dynamic shape handling policy.
    pub dynamic_shapes: DynamicShapePolicy,
}

impl CompileOptions {
    /// Creates default compilation options.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            fusion: FusionPolicy::Enabled,
            dynamic_shapes: DynamicShapePolicy::Guarded,
        }
    }
}

/// A dynamic runtime guard verifying shape and datatype expectations.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ShapeGuard {
    /// Value ID in the captured graph.
    pub value_id: ValueId,
    /// Expected tensor shape.
    pub shape: ShapeExpr,
    /// Expected logical datatype.
    pub expected_dtype: DTypeDescriptor,
}

impl ShapeGuard {
    /// Creates a new guard for a value ID with expected shape and datatype.
    #[must_use]
    pub fn new(value_id: ValueId, shape: ShapeExpr, expected_dtype: DTypeDescriptor) -> Self {
        Self {
            value_id,
            shape,
            expected_dtype,
        }
    }

    /// Verifies actual runtime shape and datatype against this guard.
    pub fn check(&self, actual_shape: &[usize], actual_dtype: DTypeDescriptor) -> Result<()> {
        if self.expected_dtype != actual_dtype {
            return Err(Error::DTypeStorageMismatch {
                expected: self.expected_dtype,
                got: actual_dtype,
            });
        }
        if let Err(reason) = self.shape.validate(actual_shape) {
            return Err(Error::Msg(alloc::format!(
                "shape guard failed for input value {}: {}",
                self.value_id,
                reason
            )));
        }
        Ok(())
    }
}

/// An immutable compiled execution plan built from a captured graph.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct CompiledPlan {
    /// The captured IR graph.
    pub graph: CapturedGraph,
    /// Compilation options used.
    pub options: CompileOptions,
    /// Dynamic shape guards for graph inputs.
    pub input_guards: Vec<ShapeGuard>,
    pub symbols: SymbolTable,
}

impl CompiledPlan {
    /// Creates a compiled plan from a captured graph and options, initializing input guards.
    #[must_use]
    pub fn compile(graph: CapturedGraph, options: CompileOptions) -> Self {
        let mut symbols = SymbolTable::default();
        let input_guards = graph
            .inputs
            .iter()
            .filter_map(|&in_id| {
                graph.value_metadata.get(&in_id).map(|value| {
                    for expr in &value.shape_expr.dims {
                        collect_symbols(expr, &mut symbols);
                    }
                    for constraint in &value.shape_expr.constraints {
                        collect_constraint_symbols(constraint, &mut symbols);
                    }
                    ShapeGuard::new(in_id, value.shape_expr.clone(), value.dtype)
                })
            })
            .collect();

        Self {
            graph,
            options,
            input_guards,
            symbols,
        }
    }

    /// Validates dynamic guards for provided inputs.
    pub fn verify_input(&self, index: usize, shape: &[usize], dtype: DTypeId) -> Result<()> {
        let guard = self.input_guards.get(index).ok_or_else(|| {
            Error::Msg(alloc::format!(
                "compiled input guard index {} is out of range ({} inputs)",
                index,
                self.input_guards.len()
            ))
        })?;
        guard.check(shape, dtype.into())
    }

    pub fn verify_inputs(&self, inputs: &[(Vec<usize>, DTypeDescriptor)]) -> Result<()> {
        if inputs.len() != self.input_guards.len() {
            return Err(Error::Msg(alloc::format!(
                "compiled invocation expected {} inputs, got {}",
                self.input_guards.len(),
                inputs.len()
            )));
        }
        let mut environment = self.symbols.environment();
        for (guard, (shape, dtype)) in self.input_guards.iter().zip(inputs) {
            if guard.expected_dtype != *dtype {
                return Err(Error::DTypeStorageMismatch {
                    expected: guard.expected_dtype,
                    got: *dtype,
                });
            }
            guard
                .shape
                .bind_and_validate(shape, &mut environment)
                .map_err(|reason| {
                    Error::Msg(alloc::format!(
                        "shape guard failed for input value {}: {}",
                        guard.value_id,
                        reason
                    ))
                })?;
        }
        let constraints = self
            .input_guards
            .iter()
            .flat_map(|guard| guard.shape.constraints.iter())
            .cloned()
            .collect::<Vec<_>>();
        environment
            .validate_constraints(&constraints)
            .map_err(|reason| {
                Error::Msg(alloc::format!("compiled invocation guard failed: {reason}"))
            })
    }
}

fn collect_symbols(expr: &crate::exec::DimExpr, symbols: &mut SymbolTable) {
    match expr {
        crate::exec::DimExpr::Symbol(id) => symbols.register(*id, None),
        crate::exec::DimExpr::Add(lhs, rhs)
        | crate::exec::DimExpr::Mul(lhs, rhs)
        | crate::exec::DimExpr::ExactDiv(lhs, rhs)
        | crate::exec::DimExpr::Broadcast(lhs, rhs) => {
            collect_symbols(lhs, symbols);
            collect_symbols(rhs, symbols);
        }
        crate::exec::DimExpr::Const(_) | crate::exec::DimExpr::Unknown => {}
    }
}

fn collect_constraint_symbols(constraint: &crate::exec::Constraint, symbols: &mut SymbolTable) {
    match constraint {
        crate::exec::Constraint::Equal { lhs, rhs }
        | crate::exec::Constraint::BroadcastCompatible { lhs, rhs } => {
            collect_symbols(lhs, symbols);
            collect_symbols(rhs, symbols);
        }
        crate::exec::Constraint::LowerBound { value, .. }
        | crate::exec::Constraint::UpperBound { value, .. }
        | crate::exec::Constraint::Divisible { value, .. } => collect_symbols(value, symbols),
    }
}
