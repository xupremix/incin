//! Immutable compiled execution plans and dynamic runtime guards.

use alloc::boxed::Box;
use alloc::string::String;
use alloc::vec::Vec;

use crate::compiled::alloc::{AllocationPlanner, LivenessMap, MemoryPlan};
use crate::compiled::capture::CapturedGraph;
use crate::exec::{DimExpr, OperationIdentity, ShapeExpr, SymbolEnvironment, SymbolTable};
use crate::graph::ValueId;
use crate::prelude::{DTypeDescriptor, DTypeId, Error, Result};

#[cfg(feature = "std")]
use crate::exec::catalog::{CapturedDescriptor, Descriptor, op};

/// Operation fusion strategy for graph compilation.
#[derive(
    Debug, Default, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize,
)]
pub enum FusionPolicy {
    /// Kernel fusion is requested, but compiled fused lowering is not yet
    /// available for this plan.
    Enabled,
    /// Kernel fusion disabled for exact execution tracing.
    #[default]
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
            fusion: FusionPolicy::Disabled,
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
    pub liveness: LivenessMap,
    pub memory_plan: MemoryPlan,
}

impl CompiledPlan {
    /// Creates a compiled plan from a captured graph and options, initializing input guards.
    pub fn compile(mut graph: CapturedGraph, options: CompileOptions) -> Result<Self> {
        if matches!(options.fusion, FusionPolicy::Enabled) {
            return Err(Error::Msg(
                "compiled fusion is not available for executable plans".into(),
            ));
        }
        for node in &graph.nodes {
            let site = node.execution_site.ok_or_else(|| {
                Error::Msg(alloc::format!(
                    "compiled operation {:?} has no execution-site classification",
                    node.operation
                ))
            })?;
            if !site.is_backend_executable() {
                return Err(Error::Msg(alloc::format!(
                    "compiled execution does not support {:?} at {:?}",
                    node.operation,
                    site
                )));
            }
        }
        propagate_symbolic_outputs(&mut graph)?;
        validate_static_constraints(&graph)?;
        let mut symbols = SymbolTable::default();
        for value in graph.value_metadata.values() {
            for constraint in &value.shape_expr.constraints {
                symbols.constraints.push(constraint.clone());
                collect_constraint_symbols(constraint, &mut symbols);
            }
        }
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
                    let shape = match options.dynamic_shapes {
                        DynamicShapePolicy::Guarded => value.shape_expr.clone(),
                        DynamicShapePolicy::Strict => ShapeExpr::concrete(&value.shape),
                    };
                    ShapeGuard::new(in_id, shape, value.dtype)
                })
            })
            .collect();

        let liveness = LivenessMap::compute(&graph);
        let memory_plan = AllocationPlanner.plan(&liveness, &graph)?;
        Ok(Self {
            graph,
            options,
            input_guards,
            symbols,
            liveness,
            memory_plan,
        })
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
            .chain(self.symbols.constraints.iter().cloned())
            .collect::<Vec<_>>();
        environment
            .validate_constraints(&constraints)
            .map_err(|reason| {
                Error::Msg(alloc::format!("compiled invocation guard failed: {reason}"))
            })
    }
}

fn validate_static_constraints(graph: &CapturedGraph) -> Result<()> {
    for value in graph.value_metadata.values() {
        for constraint in &value.shape_expr.constraints {
            let valid = match constraint {
                crate::exec::Constraint::Equal { lhs, rhs } => lhs
                    .evaluate(&[])
                    .zip(rhs.evaluate(&[]))
                    .map(|(l, r)| l == r),
                crate::exec::Constraint::LowerBound { value, bound } => {
                    value.evaluate(&[]).map(|value| value >= *bound)
                }
                crate::exec::Constraint::UpperBound { value, bound } => {
                    value.evaluate(&[]).map(|value| value <= *bound)
                }
                crate::exec::Constraint::Divisible { value, divisor } => value
                    .evaluate(&[])
                    .map(|value| *divisor != 0 && value % divisor == 0),
                crate::exec::Constraint::BroadcastCompatible { lhs, rhs } => lhs
                    .evaluate(&[])
                    .zip(rhs.evaluate(&[]))
                    .map(|(lhs, rhs)| lhs == rhs || lhs == 1 || rhs == 1),
            };
            if valid == Some(false) {
                return Err(Error::Msg(format!(
                    "compiled graph contains a statically invalid shape constraint: {constraint:?}"
                )));
            }
        }
    }
    Ok(())
}

fn propagate_symbolic_outputs(graph: &mut CapturedGraph) -> Result<()> {
    for node in &graph.nodes {
        let inputs = node
            .inputs
            .iter()
            .map(|value_id| {
                graph
                    .value_metadata
                    .get(value_id)
                    .map(|value| value.shape_expr.clone())
                    .ok_or_else(|| Error::Msg(format!("missing metadata for value {value_id}")))
            })
            .collect::<Result<Vec<_>>>()?;
        let Some(output_id) = node.outputs.first().copied() else {
            return Err(Error::Msg(format!("node {} has no output", node.id)));
        };
        let output = graph
            .value_metadata
            .get_mut(&output_id)
            .ok_or_else(|| Error::Msg(format!("missing metadata for value {output_id}")))?;
        let shape = match node.operation {
            OperationIdentity::Builtin(crate::prelude::OperationKind::Relu) => {
                inputs.first().cloned()
            }
            OperationIdentity::Builtin(
                crate::prelude::OperationKind::Abs
                | crate::prelude::OperationKind::Exp
                | crate::prelude::OperationKind::Neg
                | crate::prelude::OperationKind::Sqrt
                | crate::prelude::OperationKind::Log
                | crate::prelude::OperationKind::Tanh
                | crate::prelude::OperationKind::Sigmoid
                | crate::prelude::OperationKind::Gelu
                | crate::prelude::OperationKind::Elu
                | crate::prelude::OperationKind::Mish
                | crate::prelude::OperationKind::Swish
                | crate::prelude::OperationKind::LogicalNot,
            ) => inputs.first().cloned(),
            OperationIdentity::Builtin(
                crate::prelude::OperationKind::AddScalar
                | crate::prelude::OperationKind::SubScalar
                | crate::prelude::OperationKind::MulScalar
                | crate::prelude::OperationKind::DivScalar,
            ) => inputs.first().cloned(),
            OperationIdentity::Builtin(
                crate::prelude::OperationKind::Add
                | crate::prelude::OperationKind::Sub
                | crate::prelude::OperationKind::Mul
                | crate::prelude::OperationKind::Div,
            ) => inputs
                .first()
                .zip(inputs.get(1))
                .map(|(lhs, rhs)| broadcast_shapes(lhs, rhs)),
            OperationIdentity::Builtin(crate::prelude::OperationKind::MatMulExact) => inputs
                .first()
                .zip(inputs.get(1))
                .and_then(|(lhs, rhs)| matmul_shape(lhs, rhs)),
            OperationIdentity::Builtin(crate::prelude::OperationKind::Linear) => {
                #[cfg(feature = "std")]
                {
                    let _descriptor = decode_descriptor::<op::Linear>(node)?;
                    inputs
                        .first()
                        .zip(inputs.get(1))
                        .map(|(input, weight)| linear_shape(input, weight))
                }
                #[cfg(not(feature = "std"))]
                {
                    None
                }
            }
            OperationIdentity::Builtin(crate::prelude::OperationKind::Addmm) => inputs
                .first()
                .zip(inputs.get(1))
                .zip(inputs.get(2))
                .and_then(|((addend, lhs), rhs)| {
                    matmul_shape(lhs, rhs).map(|product| broadcast_shapes(addend, &product))
                }),
            OperationIdentity::Builtin(crate::prelude::OperationKind::ReshapeExact) => {
                #[cfg(feature = "std")]
                {
                    let descriptor = decode_descriptor::<op::ReshapeExact>(node)?;
                    Some(ShapeExpr::concrete(&descriptor.attributes().shape))
                }
                #[cfg(not(feature = "std"))]
                {
                    None
                }
            }
            OperationIdentity::Builtin(crate::prelude::OperationKind::TransposeExact) => {
                #[cfg(feature = "std")]
                {
                    let descriptor = decode_descriptor::<op::TransposeExact>(node)?;
                    inputs.first().map(|input| {
                        let mut shape = input.clone();
                        let attributes = descriptor.attributes();
                        shape.dims.swap(attributes.first, attributes.second);
                        shape
                    })
                }
                #[cfg(not(feature = "std"))]
                {
                    None
                }
            }
            OperationIdentity::Builtin(crate::prelude::OperationKind::Narrow) => {
                #[cfg(feature = "std")]
                {
                    let descriptor = decode_descriptor::<op::Narrow>(node)?;
                    inputs.first().map(|input| {
                        let mut shape = input.clone();
                        let attributes = descriptor.attributes();
                        if let Some(extent) = shape.dims.get_mut(attributes.axis) {
                            *extent = DimExpr::Const(attributes.length);
                        }
                        shape
                    })
                }
                #[cfg(not(feature = "std"))]
                {
                    None
                }
            }
            OperationIdentity::Builtin(crate::prelude::OperationKind::FlattenExact) => {
                #[cfg(feature = "std")]
                {
                    let descriptor = decode_descriptor::<op::FlattenExact>(node)?;
                    inputs.first().map(|input| {
                        let attributes = descriptor.attributes();
                        flatten_symbolic(input, attributes.start_axis, attributes.end_axis)
                    })
                }
                #[cfg(not(feature = "std"))]
                {
                    None
                }
            }
            OperationIdentity::Builtin(crate::prelude::OperationKind::SliceExact) => {
                #[cfg(feature = "std")]
                {
                    let descriptor = decode_descriptor::<op::SliceExact>(node)?;
                    inputs.first().map(|input| {
                        let mut shape = input.clone();
                        shape.dims = descriptor
                            .attributes()
                            .ranges
                            .iter()
                            .map(|(start, end)| {
                                end.checked_sub(*start)
                                    .map(DimExpr::Const)
                                    .unwrap_or(DimExpr::Unknown)
                            })
                            .collect();
                        shape
                    })
                }
                #[cfg(not(feature = "std"))]
                {
                    None
                }
            }
            OperationIdentity::Builtin(crate::prelude::OperationKind::SqueezeExact) => {
                #[cfg(feature = "std")]
                {
                    let axis = decode_descriptor::<op::SqueezeExact>(node)?
                        .attributes()
                        .axis;
                    inputs.first().map(|input| {
                        let mut shape = input.clone();
                        shape.dims.remove(axis);
                        shape.rank = crate::exec::RankExpr::Static(shape.dims.len());
                        shape
                    })
                }
                #[cfg(not(feature = "std"))]
                {
                    None
                }
            }
            OperationIdentity::Builtin(crate::prelude::OperationKind::UnsqueezeExact) => {
                #[cfg(feature = "std")]
                {
                    let axis = decode_descriptor::<op::UnsqueezeExact>(node)?
                        .attributes()
                        .axis;
                    inputs.first().map(|input| {
                        let mut shape = input.clone();
                        shape.dims.insert(axis, DimExpr::Const(1));
                        shape.rank = crate::exec::RankExpr::Static(shape.dims.len());
                        shape
                    })
                }
                #[cfg(not(feature = "std"))]
                {
                    None
                }
            }
            OperationIdentity::Builtin(crate::prelude::OperationKind::SumDim) => {
                #[cfg(feature = "std")]
                {
                    let axis = decode_descriptor::<op::SumDim>(node)?.attributes().axis;
                    inputs.first().map(|input| {
                        let mut shape = input.clone();
                        shape.dims.remove(axis);
                        shape.rank = crate::exec::RankExpr::Static(shape.dims.len());
                        shape
                    })
                }
                #[cfg(not(feature = "std"))]
                {
                    None
                }
            }
            OperationIdentity::Builtin(crate::prelude::OperationKind::SumKeepDim) => {
                #[cfg(feature = "std")]
                {
                    let axis = decode_descriptor::<op::SumKeepDim>(node)?.attributes().axis;
                    inputs.first().map(|input| {
                        let mut shape = input.clone();
                        shape.dims[axis] = DimExpr::Const(1);
                        shape
                    })
                }
                #[cfg(not(feature = "std"))]
                {
                    None
                }
            }
            OperationIdentity::Builtin(crate::prelude::OperationKind::ConcatExact) => {
                #[cfg(feature = "std")]
                {
                    let axis = decode_descriptor::<op::ConcatExact>(node)?
                        .attributes()
                        .axis;
                    inputs.first().map(|first| {
                        let mut shape = first.clone();
                        shape.dims[axis] =
                            inputs
                                .iter()
                                .skip(1)
                                .fold(shape.dims[axis].clone(), |lhs, rhs| {
                                    DimExpr::Add(Box::new(lhs), Box::new(rhs.dims[axis].clone()))
                                        .simplify()
                                });
                        shape
                    })
                }
                #[cfg(not(feature = "std"))]
                {
                    None
                }
            }
            OperationIdentity::Builtin(crate::prelude::OperationKind::StackExact) => {
                #[cfg(feature = "std")]
                {
                    let axis = decode_descriptor::<op::StackExact>(node)?.attributes().axis;
                    inputs.first().map(|input| {
                        let mut shape = input.clone();
                        shape.dims.insert(axis, DimExpr::Const(inputs.len()));
                        shape.rank = crate::exec::RankExpr::Static(shape.dims.len());
                        shape
                    })
                }
                #[cfg(not(feature = "std"))]
                {
                    None
                }
            }
            _ => None,
        };
        let Some(shape) = shape else {
            return Err(Error::Msg(format!(
                "compiled symbolic propagation does not support {:?}",
                node.operation
            )));
        };
        output.shape_expr = shape;
    }
    Ok(())
}

#[cfg(feature = "std")]
fn decode_descriptor<O>(node: &crate::compiled::CapturedNode) -> Result<Descriptor<O>>
where
    O: crate::exec::catalog::CanonicalOperation,
    O::Attributes: crate::exec::catalog::AttributeContract,
{
    let payload = node.descriptor_payload.as_ref().ok_or_else(|| {
        Error::Msg(format!(
            "compiled node {} has no captured descriptor",
            node.id
        ))
    })?;
    CapturedDescriptor::from_payload(O::ID, payload.schema, payload.payload.clone())
        .decode()
        .map_err(|error| Error::Msg(format!("invalid captured descriptor: {error}")))
}

fn broadcast_shapes(lhs: &ShapeExpr, rhs: &ShapeExpr) -> ShapeExpr {
    let rank = lhs.dims.len().max(rhs.dims.len());
    let mut dims = Vec::with_capacity(rank);
    let mut constraints = lhs.constraints.clone();
    constraints.extend(rhs.constraints.iter().cloned());
    for offset in 0..rank {
        let left = aligned_dim(&lhs.dims, rank, offset);
        let right = aligned_dim(&rhs.dims, rank, offset);
        constraints.push(crate::exec::Constraint::broadcast(
            left.clone(),
            right.clone(),
        ));
        dims.push(DimExpr::Broadcast(Box::new(left), Box::new(right)).simplify());
    }
    ShapeExpr {
        rank: crate::exec::RankExpr::Static(rank),
        dims,
        constraints,
    }
}

fn aligned_dim(dims: &[DimExpr], rank: usize, offset: usize) -> DimExpr {
    let source_offset = rank.saturating_sub(dims.len());
    if offset < source_offset {
        DimExpr::Const(1)
    } else {
        dims.get(offset - source_offset)
            .cloned()
            .unwrap_or(DimExpr::Const(1))
    }
}

fn flatten_symbolic(input: &ShapeExpr, start: usize, end: usize) -> ShapeExpr {
    let mut dims = Vec::with_capacity(input.dims.len().saturating_sub(end - start));
    dims.extend_from_slice(&input.dims[..start]);
    let flattened = input.dims[start..=end]
        .iter()
        .cloned()
        .fold(DimExpr::Const(1), |lhs, rhs| {
            DimExpr::Mul(Box::new(lhs), Box::new(rhs)).simplify()
        });
    dims.push(flattened);
    dims.extend_from_slice(&input.dims[end + 1..]);
    ShapeExpr {
        rank: crate::exec::RankExpr::Static(dims.len()),
        dims,
        constraints: input.constraints.clone(),
    }
}

fn linear_shape(input: &ShapeExpr, weight: &ShapeExpr) -> ShapeExpr {
    let mut shape = input.clone();
    if let (Some(last), Some(output)) = (shape.dims.last_mut(), weight.dims.first()) {
        let input_width = last.clone();
        *last = output.clone();
        shape.constraints.push(crate::exec::Constraint::equal(
            input_width,
            weight.dims.get(1).cloned().unwrap_or(DimExpr::Unknown),
        ));
    }
    shape
}

fn matmul_shape(lhs: &ShapeExpr, rhs: &ShapeExpr) -> Option<ShapeExpr> {
    if lhs.dims.len() < 2 || rhs.dims.len() < 2 {
        return None;
    }
    let left_batch = ShapeExpr {
        rank: crate::exec::RankExpr::Static(lhs.dims.len() - 2),
        dims: lhs.dims[..lhs.dims.len() - 2].to_vec(),
        constraints: lhs.constraints.clone(),
    };
    let right_batch = ShapeExpr {
        rank: crate::exec::RankExpr::Static(rhs.dims.len() - 2),
        dims: rhs.dims[..rhs.dims.len() - 2].to_vec(),
        constraints: rhs.constraints.clone(),
    };
    let batches = broadcast_shapes(&left_batch, &right_batch);
    let mut result = batches.dims;
    result.push(lhs.dims[lhs.dims.len() - 2].clone());
    result.push(rhs.dims[rhs.dims.len() - 1].clone());
    Some(ShapeExpr {
        rank: crate::exec::RankExpr::Static(result.len()),
        dims: result,
        constraints: batches
            .constraints
            .into_iter()
            .chain(core::iter::once(crate::exec::Constraint::equal(
                lhs.dims[lhs.dims.len() - 1].clone(),
                rhs.dims[rhs.dims.len() - 2].clone(),
            )))
            .collect(),
    })
}

fn collect_symbols(expr: &crate::exec::DimExpr, symbols: &mut SymbolTable) {
    match expr {
        crate::exec::DimExpr::Symbol(id) => symbols.register(*id, None, None),
        crate::exec::DimExpr::NamedSymbol { id, name, identity } => {
            symbols.register(*id, Some(name.clone()), Some(identity.clone()));
        }
        crate::exec::DimExpr::Add(lhs, rhs)
        | crate::exec::DimExpr::Sub(lhs, rhs)
        | crate::exec::DimExpr::Mul(lhs, rhs)
        | crate::exec::DimExpr::ExactDiv(lhs, rhs)
        | crate::exec::DimExpr::Broadcast(lhs, rhs)
        | crate::exec::DimExpr::Min(lhs, rhs)
        | crate::exec::DimExpr::Max(lhs, rhs) => {
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
