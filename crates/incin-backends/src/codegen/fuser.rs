//! Legality-checked pointwise fuser: a proven CMP-005 group becomes one kernel.
//!
//! Strategy 1 from `docs/plan/research/0.2.0/kernel-generation-strategies.md`:
//! a fusion group may only grow when it cuts reads/writes and every member is
//! elementwise-class. `incin_core::compiled::fusion` proves that shape of group
//! (exclusive intermediates, tape check, topological order, one geometry);
//! this module turns such a group into a single scalar expression and drops it
//! into the existing `kernel::scalar` templates via `lower_scalar`, so strides,
//! packing, autotune keys and NVRTC compilation are inherited unchanged.
//!
//! Every entry point fails closed: a step outside the CMP-005 allow-list, a
//! boundary the two-operand templates cannot hold, a saved intermediate, or a
//! shape/dtype disagreement is an error, never a best-effort kernel.
#![allow(dead_code)]

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use incin_core::error::{Error, Result};
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::dtype::{DTypeDescriptor, DTypeId};

use super::catalog::{binary_forward, unary_forward};
use super::ir::IrExpr;

#[cfg(feature = "compiled")]
use alloc::collections::BTreeMap;
#[cfg(feature = "compiled")]
use incin_core::compiled::{CapturedGraph, FusedKernel, SavedTensorSet};
#[cfg(feature = "compiled")]
use incin_core::exec::OperationIdentity;
#[cfg(feature = "compiled")]
use incin_core::graph::ValueId;

/// The argument slot that stands for the previous step's output while a step
/// is built in isolation. Real boundaries only ever occupy `Arg(0)`/`Arg(1)`
/// because a chain has at most two operands, so `Arg(2)` cannot collide.
const SPINE_ARG: usize = 2;

/// Where a chain step reads one of its inputs from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChainOperandRef {
    /// The previous step's output — the chain spine. Illegal on step 0.
    Spine,
    /// A boundary operand, by index into [`PointwiseChain::operands`].
    Boundary(usize),
}

/// One boundary value the emitted kernel loads from the outside.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ChainOperand {
    /// Graph `ValueId` at the boundary (synthetic ids in hand-built chains).
    pub(crate) id: usize,
    /// Shape of the boundary value; must equal [`PointwiseChain::shape`].
    pub(crate) shape: Vec<usize>,
    /// Dtype of the boundary value; must equal [`PointwiseChain::dtype`].
    pub(crate) dtype: DTypeDescriptor,
}

/// One elementwise operation in the chain, with input wiring.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ChainStep {
    /// The operation, restricted at build time to the CMP-005 allow-list.
    pub(crate) op: OperationKind,
    /// One reference per catalog arity position.
    pub(crate) inputs: Vec<ChainOperandRef>,
}

/// A proven pointwise chain ready to lower into one kernel body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PointwiseChain {
    /// Ordered steps; the spine wires step *k−1*'s output into step *k*.
    pub(crate) steps: Vec<ChainStep>,
    /// Distinct external values, in first-appearance order. 1 → unary
    /// template (`x`), 2 → binary template (`a`, `b`).
    pub(crate) operands: Vec<ChainOperand>,
    /// Shape every value in the chain was proven to share.
    pub(crate) shape: Vec<usize>,
    /// Dtype every value in the chain was proven to share.
    pub(crate) dtype: DTypeId,
}

/// The lowered fused expression plus the identity the renderer needs.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FusedChainExpr {
    /// Cache/entry identifier: `fused` + each step's catalog name.
    pub(crate) op_name: String,
    /// Expression over `Arg(0)`/`Arg(1)` boundary slots only.
    pub(crate) expr: IrExpr,
    /// 1 or 2 — selects the unary or binary template.
    pub(crate) operand_count: usize,
    /// Storage dtype of the chain (half types compute as `float`).
    pub(crate) dtype: DTypeId,
}

/// Catalog arity for a CMP-005 pointwise operation, or `None` when the op is
/// not in the allow-list (including every non-elementwise identity).
const fn op_arity(op: OperationKind) -> Option<usize> {
    use OperationKind::*;
    match op {
        Add | Sub | Mul | Div => Some(2),
        Relu | Gelu | Sigmoid | Tanh | Swish | Neg | Abs | Exp | Sqrt | Log => Some(1),
        _ => None,
    }
}

/// The catalog forward template for `op`, already expressed over `Arg(i)`.
fn catalog_template(op: OperationKind) -> Option<IrExpr> {
    match op_arity(op)? {
        1 => unary_forward(op.name()),
        _ => binary_forward(op.name()),
    }
}

/// Replaces every `Arg(index)` in `expr` with `replacement`.
fn substitute_arg(expr: &IrExpr, index: usize, replacement: &IrExpr) -> IrExpr {
    match expr {
        IrExpr::Arg(i) if *i == index => replacement.clone(),
        IrExpr::Arg(_) | IrExpr::Const(_) | IrExpr::Var(_) => expr.clone(),
        IrExpr::Unary(op, inner) => IrExpr::unary(*op, substitute_arg(inner, index, replacement)),
        IrExpr::Binary(op, lhs, rhs) => IrExpr::binary(
            *op,
            substitute_arg(lhs, index, replacement),
            substitute_arg(rhs, index, replacement),
        ),
        IrExpr::Ternary(op, a, b, c) => IrExpr::ternary(
            *op,
            substitute_arg(a, index, replacement),
            substitute_arg(b, index, replacement),
            substitute_arg(c, index, replacement),
        ),
    }
}

/// Whether `expr` still references `Arg(index)`.
fn contains_arg(expr: &IrExpr, index: usize) -> bool {
    match expr {
        IrExpr::Arg(i) => *i == index,
        IrExpr::Const(_) | IrExpr::Var(_) => false,
        IrExpr::Unary(_, inner) => contains_arg(inner, index),
        IrExpr::Binary(_, lhs, rhs) => contains_arg(lhs, index) || contains_arg(rhs, index),
        IrExpr::Ternary(_, a, b, c) => {
            contains_arg(a, index) || contains_arg(b, index) || contains_arg(c, index)
        }
    }
}

impl PointwiseChain {
    /// Identifier under which the rendered kernel is cached and launched.
    pub(crate) fn op_name(&self) -> String {
        let mut name = String::from("fused");
        for step in &self.steps {
            name.push('_');
            name.push_str(step.op.name());
        }
        name
    }

    /// Lowers the chain to one expression over the boundary operands.
    ///
    /// Fail-closed gates, in order:
    ///
    /// 1. at least two steps — a single op is not a fusion;
    /// 2. one or two boundary operands — the templates' arity;
    /// 3. a float storage dtype (`F16`/`BF16`/`F32`/`F64`);
    /// 4. every operand shares the chain's shape and dtype;
    /// 5. every step is in the CMP-005 allow-list with catalog arity and a
    ///    catalog lowering, wires exactly that many inputs, reads the spine
    ///    from step 1 on (and never on step 0), and indexes boundaries in
    ///    range.
    ///
    /// # Errors
    ///
    /// Returns the first gate the chain fails.
    pub(crate) fn build_expr(&self) -> Result<IrExpr> {
        if self.steps.len() < 2 {
            return Err(Error::Msg("fused chain requires at least two steps".into()));
        }
        let operand_count = self.operands.len();
        if !(1..=2).contains(&operand_count) {
            return Err(Error::Msg(format!(
                "fused chain requires one or two boundary operands, got {operand_count}"
            )));
        }
        if !matches!(
            self.dtype,
            DTypeId::F16 | DTypeId::BF16 | DTypeId::F32 | DTypeId::F64
        ) {
            return Err(Error::Msg(format!(
                "fused chain requires a float dtype, got {:?}",
                self.dtype
            )));
        }
        for (index, operand) in self.operands.iter().enumerate() {
            if operand.shape != self.shape {
                return Err(Error::Msg(format!(
                    "fused chain operand {index} shape {:?} disagrees with chain shape {:?}",
                    operand.shape, self.shape
                )));
            }
            if operand.dtype.builtin_id() != Some(self.dtype) {
                return Err(Error::Msg(format!(
                    "fused chain operand {index} dtype {:?} disagrees with chain dtype {:?}",
                    operand.dtype, self.dtype
                )));
            }
        }

        let mut current: Option<IrExpr> = None;
        for (step_index, step) in self.steps.iter().enumerate() {
            let arity = op_arity(step.op).ok_or_else(|| {
                Error::Msg(format!(
                    "fused chain step {step_index} op {} is not a fusable pointwise operation",
                    step.op.name()
                ))
            })?;
            if step.inputs.len() != arity {
                return Err(Error::Msg(format!(
                    "fused chain step {step_index} ({}) expects {arity} input(s), got {}",
                    step.op.name(),
                    step.inputs.len()
                )));
            }
            let mut reads_spine = false;
            for input in &step.inputs {
                match input {
                    ChainOperandRef::Spine => {
                        if step_index == 0 {
                            return Err(Error::Msg(
                                "fused chain step 0 cannot read the spine".into(),
                            ));
                        }
                        reads_spine = true;
                    }
                    ChainOperandRef::Boundary(boundary) => {
                        if *boundary >= operand_count {
                            return Err(Error::Msg(format!(
                                "fused chain step {step_index} references boundary {boundary} \
                                 but the chain has {operand_count} operand(s)"
                            )));
                        }
                    }
                }
            }
            if step_index > 0 && !reads_spine {
                return Err(Error::Msg(format!(
                    "fused chain step {step_index} does not read the previous step's output"
                )));
            }

            let template = catalog_template(step.op).ok_or_else(|| {
                Error::Msg(format!(
                    "fused chain step {step_index} op {} has no catalog lowering",
                    step.op.name()
                ))
            })?;
            // Catalog templates for the allow-list reference exactly
            // `Arg(0..arity)`, and `inputs.len() == arity` was checked above,
            // so every index the closure sees is in range.
            let inputs = &step.inputs;
            let mapped = template.remap_args(&|index| match inputs[index] {
                ChainOperandRef::Boundary(boundary) => boundary,
                ChainOperandRef::Spine => SPINE_ARG,
            });

            let step_expr = match current.take() {
                None => {
                    if contains_arg(&mapped, SPINE_ARG) {
                        return Err(Error::Msg("fused chain step 0 references the spine".into()));
                    }
                    mapped
                }
                Some(previous) => substitute_arg(&mapped, SPINE_ARG, &previous),
            };
            current = Some(step_expr);
        }
        current.ok_or_else(|| Error::Msg("fused chain has no steps".into()))
    }

    /// Lowers the chain and packages it for the renderer.
    ///
    /// # Errors
    ///
    /// Returns the first legality gate [`Self::build_expr`] refuses.
    pub(crate) fn fused(&self) -> Result<FusedChainExpr> {
        let expr = self.build_expr()?;
        Ok(FusedChainExpr {
            op_name: self.op_name(),
            expr,
            operand_count: self.operands.len(),
            dtype: self.dtype,
        })
    }
}

/// Rebuilds a planned group from a captured graph, re-validating every gate.
///
/// The group is re-checked rather than trusted: each consecutive link runs
/// [`incin_core::compiled::FusionPass::check_link`] again, every intermediate
/// must be absent from `saved`, all path values must share one shape and
/// builtin dtype, and the boundary set must fit the two-operand templates.
///
/// # Errors
///
/// Returns an error when any re-validation gate fails.
#[cfg(feature = "compiled")]
pub(crate) fn chain_from_graph(
    graph: &CapturedGraph,
    group: &FusedKernel,
    saved: &SavedTensorSet,
) -> Result<PointwiseChain> {
    let indices = &group.source_node_indices;
    if indices.len() < 2 {
        return Err(Error::Msg("fused group requires at least two nodes".into()));
    }
    for window in indices.windows(2) {
        incin_core::compiled::FusionPass
            .check_link(graph, window[0], window[1])
            .map_err(|blocker| {
                Error::Msg(format!(
                    "fused group link {}->{} refused: {blocker:?}",
                    window[0], window[1]
                ))
            })?;
    }
    for &index in indices.iter().take(indices.len() - 1) {
        let intermediate = graph.nodes[index].outputs[0];
        if saved.contains(intermediate) {
            return Err(Error::Msg(format!(
                "fused group intermediate {intermediate} is saved for backward"
            )));
        }
    }

    let mut reference: Option<(Vec<usize>, DTypeId)> = None;
    for &index in indices {
        let node = &graph.nodes[index];
        for &value in node.inputs.iter().chain(node.outputs.iter()) {
            let meta = graph
                .value_metadata
                .get(&value)
                .ok_or_else(|| Error::Msg(format!("fused group value {value} has no metadata")))?;
            let dtype = meta.dtype.builtin_id().ok_or_else(|| {
                Error::Msg(format!("fused group value {value} has a custom dtype"))
            })?;
            match &reference {
                None => reference = Some((meta.shape.clone(), dtype)),
                Some((shape, known)) => {
                    if *shape != meta.shape || *known != dtype {
                        return Err(Error::Msg(format!(
                            "fused group value {value} disagrees with the chain's shape/dtype"
                        )));
                    }
                }
            }
        }
    }
    let (shape, dtype) = reference.ok_or_else(|| Error::Msg("fused group is empty".into()))?;

    let mut boundary_index: BTreeMap<ValueId, usize> = BTreeMap::new();
    let mut operands: Vec<ChainOperand> = Vec::new();
    let mut steps: Vec<ChainStep> = Vec::new();

    for (step_index, &index) in indices.iter().enumerate() {
        let node = &graph.nodes[index];
        let op = match &node.operation {
            OperationIdentity::Builtin(op) => *op,
            OperationIdentity::Custom(_) => {
                return Err(Error::Msg(format!(
                    "fused group step {step_index} is a custom operation"
                )));
            }
        };
        let spine = (step_index > 0).then(|| graph.nodes[indices[step_index - 1]].outputs[0]);
        let mut inputs = Vec::with_capacity(node.inputs.len());
        for &input in &node.inputs {
            if spine == Some(input) {
                inputs.push(ChainOperandRef::Spine);
                continue;
            }
            let slot = match boundary_index.get(&input) {
                Some(existing) => *existing,
                None => {
                    let slot = operands.len();
                    let meta = graph.value_metadata.get(&input).ok_or_else(|| {
                        Error::Msg(format!("fused group boundary {input} has no metadata"))
                    })?;
                    operands.push(ChainOperand {
                        id: input,
                        shape: meta.shape.clone(),
                        dtype: meta.dtype,
                    });
                    boundary_index.insert(input, slot);
                    slot
                }
            };
            inputs.push(ChainOperandRef::Boundary(slot));
        }
        steps.push(ChainStep { op, inputs });
    }

    if operands.is_empty() || operands.len() > 2 {
        return Err(Error::Msg(format!(
            "fused group has {} boundary operand(s); the pointwise templates support one or two",
            operands.len()
        )));
    }

    Ok(PointwiseChain {
        steps,
        operands,
        shape,
        dtype,
    })
}

/// Renders a fused chain through the existing scalar templates.
///
/// The fragment is lowered in the chain's own dtype: `lower_scalar` already
/// resolves `F16`/`BF16` to a `float` compute type, matching what
/// `CudaScalarSpec` declares for those storages.
///
/// # Errors
///
/// Returns an error when the chain refuses a legality gate or the underlying
/// template renderer fails.
#[cfg(any(
    feature = "cuda",
    all(test, any(feature = "metal", feature = "wgpu", feature = "autotune"))
))]
pub(crate) fn render_fused_kernel(
    chain: &PointwiseChain,
    layout: incin_core::exec::LayoutClass,
    unroll_width: u8,
) -> Result<crate::kernel::RenderedKernel> {
    let fused = chain.fused()?;
    let fragment = super::lower_scalar(
        &fused.expr,
        match fused.operand_count {
            1 => &["x"],
            _ => &["a", "b"],
        },
        fused.dtype,
    )?;
    match fused.operand_count {
        1 => crate::kernel::render_cuda_unary_for_layout_body(
            &fused.op_name,
            &fragment,
            fused.dtype,
            layout,
            unroll_width,
            None,
        ),
        2 => crate::kernel::render_cuda_binary_for_layout_body(
            &fused.op_name,
            &fragment,
            fused.dtype,
            layout,
            unroll_width,
        ),
        other => Err(Error::Msg(format!(
            "fused chain rendered with unsupported operand count {other}"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{CpuJitKernel, KernelDefinition};

    fn operand(id: usize) -> ChainOperand {
        ChainOperand {
            id,
            shape: alloc::vec![4],
            dtype: DTypeId::F32.descriptor(),
        }
    }

    fn f32_chain(steps: Vec<ChainStep>, operand_count: usize) -> PointwiseChain {
        PointwiseChain {
            steps,
            operands: (0..operand_count).map(operand).collect(),
            shape: alloc::vec![4],
            dtype: DTypeId::F32,
        }
    }

    /// `out = mul(relu(x), c)` — the two-boundary workhorse.
    fn relu_mul_chain() -> PointwiseChain {
        f32_chain(
            alloc::vec![
                ChainStep {
                    op: OperationKind::Relu,
                    inputs: alloc::vec![ChainOperandRef::Boundary(0)],
                },
                ChainStep {
                    op: OperationKind::Mul,
                    inputs: alloc::vec![ChainOperandRef::Spine, ChainOperandRef::Boundary(1)],
                },
            ],
            2,
        )
    }

    /// `out = abs(neg(x))` — one boundary, unary template.
    fn neg_abs_chain() -> PointwiseChain {
        f32_chain(
            alloc::vec![
                ChainStep {
                    op: OperationKind::Neg,
                    inputs: alloc::vec![ChainOperandRef::Boundary(0)],
                },
                ChainStep {
                    op: OperationKind::Abs,
                    inputs: alloc::vec![ChainOperandRef::Spine],
                },
            ],
            1,
        )
    }

    #[test]
    fn op_name_joins_each_step_catalog_name() {
        assert_eq!(relu_mul_chain().op_name(), "fused_relu_mul");
        assert_eq!(neg_abs_chain().op_name(), "fused_neg_abs");
    }

    #[test]
    fn fused_outputs_match_unfused_catalog_composition() {
        let chain = relu_mul_chain();
        let expr = chain.build_expr().expect("chain should lower");
        let relu = unary_forward("relu").expect("relu is in the catalog");
        let mul = binary_forward("mul").expect("mul is in the catalog");

        for &x in &[-2.0, -0.5, 0.0, 1.5, 3.0] {
            for &k in &[0.5, 2.0, -1.0] {
                let fused = expr.eval(&[x, k]);
                let unfused = mul.eval(&[relu.eval(&[x]), k]);
                assert!(
                    (fused - unfused).abs() < 1e-12,
                    "fused({x}, {k}) = {fused}, unfused = {unfused}"
                );
            }
        }
    }

    #[test]
    fn fused_gradients_match_chain_rule_away_from_the_relu_kink() {
        let chain = relu_mul_chain();
        let expr = chain.build_expr().expect("chain should lower");
        let definition = KernelDefinition::new("fused_relu_mul", 2, DTypeId::F32, expr.clone());
        let d_dx = &definition.backward_derivatives[0];

        let h = 1e-6;
        for &x in &[0.7, 2.0, -1.3, -0.25] {
            for &k in &[0.5, 2.0] {
                let analytical = d_dx.eval(&[x, k]);
                let expected = if x > 0.0 { k } else { 0.0 };
                assert!(
                    (analytical - expected).abs() < 1e-12,
                    "chain rule at x={x}, k={k}: got {analytical}, want {expected}"
                );
                let f = |v: f64| expr.eval(&[v, k]);
                let numerical = (f(x + h) - f(x - h)) / (2.0 * h);
                assert!(
                    (analytical - numerical).abs() < 1e-4,
                    "central difference at x={x}, k={k}: analytical {analytical} vs {numerical}"
                );
            }
        }
    }

    #[test]
    fn cpu_jit_forward_and_backward_match_stepwise_execution() {
        let chain = relu_mul_chain();
        let fused_expr = chain.build_expr().expect("chain should lower");
        let definition =
            KernelDefinition::new("fused_relu_mul", 2, DTypeId::F32, fused_expr.clone());
        let kernel = CpuJitKernel::new(definition);

        let xs = [1.0f32, -2.0, 0.5, -0.25];
        let cs = [2.0f32, 3.0, -1.0, 0.75];
        let mut out = [0.0f32; 4];
        kernel
            .eval_f32(&[&xs, &cs], &mut out)
            .expect("fused forward should evaluate");
        for i in 0..xs.len() {
            let unfused = xs[i].max(0.0) * cs[i];
            assert!(
                (out[i] - unfused).abs() < 1e-6,
                "forward[{i}]: fused {} vs unfused {unfused}",
                out[i]
            );
        }

        let grad_out = [1.0f32, 1.0, 1.0, 1.0];
        let mut grad_x = [0.0f32; 4];
        kernel
            .eval_backward_f32(&grad_out, &[&xs, &cs], 0, &mut grad_x)
            .expect("fused backward should evaluate");
        for i in 0..xs.len() {
            let chain_rule = if xs[i] > 0.0 { cs[i] } else { 0.0 };
            assert!(
                (grad_x[i] - chain_rule).abs() < 1e-6,
                "backward[{i}]: fused {} vs chain rule {chain_rule}",
                grad_x[i]
            );
        }
    }

    #[test]
    fn single_step_is_refused() {
        let chain = f32_chain(
            alloc::vec![ChainStep {
                op: OperationKind::Relu,
                inputs: alloc::vec![ChainOperandRef::Boundary(0)],
            }],
            1,
        );
        let err = chain.build_expr().expect_err("one step is not a fusion");
        assert!(err.to_string().contains("at least two steps"), "{err}");
    }

    #[test]
    fn three_boundaries_are_refused() {
        let chain = f32_chain(
            alloc::vec![
                ChainStep {
                    op: OperationKind::Add,
                    inputs: alloc::vec![ChainOperandRef::Boundary(0), ChainOperandRef::Boundary(1),],
                },
                ChainStep {
                    op: OperationKind::Mul,
                    inputs: alloc::vec![ChainOperandRef::Spine, ChainOperandRef::Boundary(2)],
                },
            ],
            3,
        );
        let err = chain.build_expr().expect_err("templates hold two operands");
        assert!(err.to_string().contains("one or two"), "{err}");
    }

    #[test]
    fn matmul_step_is_refused() {
        let chain = f32_chain(
            alloc::vec![
                ChainStep {
                    op: OperationKind::MatMul,
                    inputs: alloc::vec![ChainOperandRef::Boundary(0), ChainOperandRef::Boundary(1),],
                },
                ChainStep {
                    op: OperationKind::Relu,
                    inputs: alloc::vec![ChainOperandRef::Spine],
                },
            ],
            2,
        );
        let err = chain.build_expr().expect_err("matmul is not elementwise");
        assert!(
            err.to_string()
                .contains("not a fusable pointwise operation"),
            "{err}"
        );
    }

    #[test]
    fn operand_shape_disagreeing_with_the_chain_is_refused() {
        let mut chain = relu_mul_chain();
        chain.operands[1].shape = alloc::vec![2];
        let err = chain.build_expr().expect_err("shapes must match");
        assert!(err.to_string().contains("disagrees"), "{err}");
    }

    #[test]
    fn integer_dtype_is_refused() {
        let mut chain = relu_mul_chain();
        chain.dtype = DTypeId::I64;
        let err = chain.build_expr().expect_err("fusion is float-only");
        assert!(err.to_string().contains("float dtype"), "{err}");
    }

    #[test]
    fn spine_reference_on_step_zero_is_refused() {
        let chain = f32_chain(
            alloc::vec![
                ChainStep {
                    op: OperationKind::Neg,
                    inputs: alloc::vec![ChainOperandRef::Spine],
                },
                ChainStep {
                    op: OperationKind::Abs,
                    inputs: alloc::vec![ChainOperandRef::Spine],
                },
            ],
            1,
        );
        let err = chain.build_expr().expect_err("step 0 has no spine");
        assert!(err.to_string().contains("step 0"), "{err}");
    }

    #[test]
    fn boundary_index_out_of_range_is_refused() {
        let chain = f32_chain(
            alloc::vec![
                ChainStep {
                    op: OperationKind::Relu,
                    inputs: alloc::vec![ChainOperandRef::Boundary(0)],
                },
                ChainStep {
                    op: OperationKind::Neg,
                    inputs: alloc::vec![ChainOperandRef::Spine],
                },
                ChainStep {
                    op: OperationKind::Add,
                    inputs: alloc::vec![ChainOperandRef::Spine, ChainOperandRef::Boundary(1),],
                },
            ],
            1,
        );
        let err = chain.build_expr().expect_err("boundary 1 does not exist");
        assert!(err.to_string().contains("boundary 1"), "{err}");
    }

    #[test]
    fn arity_mismatch_is_refused() {
        let chain = f32_chain(
            alloc::vec![
                ChainStep {
                    op: OperationKind::Relu,
                    inputs: alloc::vec![ChainOperandRef::Boundary(0)],
                },
                ChainStep {
                    op: OperationKind::Mul,
                    inputs: alloc::vec![ChainOperandRef::Spine],
                },
            ],
            1,
        );
        let err = chain.build_expr().expect_err("mul needs two inputs");
        assert!(err.to_string().contains("expects 2 input"), "{err}");
    }

    #[test]
    fn disconnected_step_that_skips_the_spine_is_refused() {
        let chain = f32_chain(
            alloc::vec![
                ChainStep {
                    op: OperationKind::Relu,
                    inputs: alloc::vec![ChainOperandRef::Boundary(0)],
                },
                ChainStep {
                    op: OperationKind::Neg,
                    inputs: alloc::vec![ChainOperandRef::Boundary(0)],
                },
            ],
            1,
        );
        let err = chain
            .build_expr()
            .expect_err("step 1 must consume step 0's output");
        assert!(err.to_string().contains("previous step's output"), "{err}");
    }

    #[cfg(all(
        test,
        any(
            feature = "cuda",
            feature = "metal",
            feature = "wgpu",
            feature = "autotune"
        )
    ))]
    mod render {
        use super::*;
        use incin_core::exec::LayoutClass;

        #[test]
        fn unary_fused_kernel_emits_one_store_and_an_ssa_prologue() {
            let rendered = render_fused_kernel(&neg_abs_chain(), LayoutClass::Contiguous, 1)
                .expect("unary chain should render");
            assert_eq!(
                rendered.source.matches("output[idx] =").count(),
                1,
                "{}",
                rendered.source
            );
            assert!(
                rendered.source.contains("const float t"),
                "SSA prologue missing:\n{}",
                rendered.source
            );
            assert!(rendered.entry_point.contains("fused_neg_abs"));
        }

        #[test]
        fn distinct_chains_get_distinct_cache_keys() {
            let relu_add = f32_chain(
                alloc::vec![
                    ChainStep {
                        op: OperationKind::Relu,
                        inputs: alloc::vec![ChainOperandRef::Boundary(0)],
                    },
                    ChainStep {
                        op: OperationKind::Add,
                        inputs: alloc::vec![ChainOperandRef::Spine, ChainOperandRef::Boundary(1)],
                    },
                ],
                2,
            );
            let relu_mul = relu_mul_chain();
            let a = render_fused_kernel(&relu_mul, LayoutClass::Contiguous, 1)
                .expect("relu_mul should render");
            let b = render_fused_kernel(&relu_add, LayoutClass::Contiguous, 1)
                .expect("relu_add should render");
            assert_ne!(a.entry_point, b.entry_point);
            assert_ne!(a.cache_key, b.cache_key);
        }
    }

    #[cfg(all(test, feature = "compiled"))]
    mod from_graph {
        use super::*;
        use alloc::collections::BTreeMap;
        use incin_core::graph::Graph;
        use incin_core::prelude::OperationKind as PreludeOp;

        // `DTypeId` is already in scope from the parent; alias only what the
        // graph fixture needs that might collide.
        use incin_core::prelude::DTypeId;

        fn relu_mul_graph() -> (CapturedGraph, ValueId, ValueId, ValueId) {
            let mut graph = Graph::new();
            let x = graph.add_value(alloc::vec![4], DTypeId::F32, Some("x".into()));
            let c = graph.add_value(alloc::vec![4], DTypeId::F32, Some("c".into()));
            let y = graph.add_value(alloc::vec![4], DTypeId::F32, Some("y".into()));
            let out = graph.add_value(alloc::vec![4], DTypeId::F32, Some("out".into()));
            graph.mark_input(x);
            graph.mark_input(c);
            graph.mark_output(out);
            graph.add_node(
                PreludeOp::Relu,
                alloc::vec![x],
                alloc::vec![y],
                BTreeMap::new(),
            );
            graph.add_node(
                PreludeOp::Mul,
                alloc::vec![y, c],
                alloc::vec![out],
                BTreeMap::new(),
            );
            let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
            (captured, x, c, y)
        }

        #[test]
        fn chain_from_graph_rebuilds_a_proven_two_node_group() {
            let (captured, x, c, _y) = relu_mul_graph();
            let group = FusedKernel {
                source_node_indices: alloc::vec![0, 1],
                primary_op: PreludeOp::Relu,
            };
            let chain = chain_from_graph(&captured, &group, &SavedTensorSet::new())
                .expect("proven group should lower");
            assert_eq!(chain.steps.len(), 2);
            assert_eq!(chain.steps[0].op, PreludeOp::Relu);
            assert_eq!(chain.steps[1].op, PreludeOp::Mul);
            assert_eq!(
                chain.steps[0].inputs,
                alloc::vec![ChainOperandRef::Boundary(0)]
            );
            assert_eq!(
                chain.steps[1].inputs,
                alloc::vec![ChainOperandRef::Spine, ChainOperandRef::Boundary(1)]
            );
            let ids: alloc::vec::Vec<usize> =
                chain.operands.iter().map(|operand| operand.id).collect();
            assert_eq!(ids, alloc::vec![x, c]);
            assert_eq!(chain.dtype, DTypeId::F32);
            assert_eq!(chain.shape, alloc::vec![4]);
            let fused = chain.fused().expect("chain should lower");
            assert_eq!(fused.op_name, "fused_relu_mul");
            assert_eq!(fused.operand_count, 2);
        }

        #[test]
        fn chain_from_graph_refuses_a_saved_intermediate() {
            let (captured, _x, _c, y) = relu_mul_graph();
            let group = FusedKernel {
                source_node_indices: alloc::vec![0, 1],
                primary_op: PreludeOp::Relu,
            };
            let mut saved = SavedTensorSet::new();
            saved.save(y);
            let err = chain_from_graph(&captured, &group, &saved)
                .expect_err("saved intermediates cannot disappear");
            assert!(err.to_string().contains("saved for backward"), "{err}");
        }

        #[test]
        fn chain_from_graph_refuses_a_non_elementwise_link() {
            let mut graph = Graph::new();
            let a = graph.add_value(alloc::vec![4], DTypeId::F32, Some("a".into()));
            let b = graph.add_value(alloc::vec![4], DTypeId::F32, Some("b".into()));
            let y = graph.add_value(alloc::vec![4], DTypeId::F32, Some("y".into()));
            let out = graph.add_value(alloc::vec![4], DTypeId::F32, Some("out".into()));
            graph.mark_input(a);
            graph.mark_input(b);
            graph.mark_output(out);
            graph.add_node(
                PreludeOp::MatMul,
                alloc::vec![a, b],
                alloc::vec![y],
                BTreeMap::new(),
            );
            graph.add_node(
                PreludeOp::Relu,
                alloc::vec![y],
                alloc::vec![out],
                BTreeMap::new(),
            );
            let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
            let group = FusedKernel {
                source_node_indices: alloc::vec![0, 1],
                primary_op: PreludeOp::MatMul,
            };
            let err = chain_from_graph(&captured, &group, &SavedTensorSet::new())
                .expect_err("matmul must not enter a pointwise chain");
            assert!(err.to_string().contains("refused"), "{err}");
        }
    }

    #[cfg(all(test, feature = "cuda"))]
    mod dispatch {
        use super::*;
        use incin_core::exec::LayoutClass;

        /// Compile-and-load the rendered fused kernel on device 0.
        ///
        /// Ignored in normal runs: it needs real CUDA hardware. It certifies
        /// that the rendered source is valid NVRTC C and that the entry point
        /// resolves — it does not launch and does not compare device output
        /// against a host reference.
        #[test]
        #[ignore = "requires CUDA hardware"]
        fn fused_kernel_compiles_and_resolves_on_device() {
            let rendered = render_fused_kernel(&relu_mul_chain(), LayoutClass::Contiguous, 1)
                .expect("chain should render");
            let dispatcher = crate::cuda::gpu::CpuCudaDispatcher::new(0)
                .expect("CUDA device 0 should initialize");
            dispatcher
                .compile_and_load_kernel(
                    &rendered.entry_point,
                    &rendered.source,
                    "incin_fuser_smoke",
                )
                .expect("fused source should compile under NVRTC");
            let _function = dispatcher
                .get_function("incin_fuser_smoke", &rendered.entry_point)
                .expect("entry point should resolve");
        }
    }
}
