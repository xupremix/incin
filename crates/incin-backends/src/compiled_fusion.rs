//! Executable lowering of proven CMP-005 fusion groups: one group, one kernel.
//!
//! Core's `FusionPass::apply` proves a pointwise chain exclusive, tape-safe,
//! geometrically uniform, float typed, and within the template boundary
//! budget, then admits it as a `FusedKernel` — still only node indices in a
//! `CapturedGraph`. [`PointwiseScalarLowering`] implements the core
//! `FusedKernelLowering` contract for this crate: the group is rebuilt and
//! re-validated from the graph through the crate-private `codegen::fuser`
//! chain builder (link gates, tape, geometry, boundaries), lowered to a
//! single composed `IrExpr`, and packaged as one `KernelDefinition` whose
//! symbolic derivatives cover every boundary operand.
//!
//! Every gate fails closed: a group that bypassed admission, a saved
//! intermediate, three boundary operands, a non-elementwise link, or a
//! non-float dtype is an error, never a partial kernel.

use incin_core::compiled::{CapturedGraph, FusedKernel, FusedKernelLowering, SavedTensorSet};
use incin_core::error::Result;

use crate::codegen::KernelDefinition;

/// Lowers an admitted group through the scalar-catalog fuser into one kernel.
///
/// The stock `FusedKernelLowering` implementation for every `compiled` build.
/// The emitted [`KernelDefinition`] is the single executable unit: its
/// `forward` is the whole chain composed into one expression over the
/// boundary operands, and its `backward_derivatives` are the symbolic diffs
/// of that composition, so a CPU-JIT reference run, the CUDA JIT path, and
/// the scalar renderers all consume the same fused artifact.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PointwiseScalarLowering;

impl FusedKernelLowering for PointwiseScalarLowering {
    type Kernel = KernelDefinition;

    fn lower_group(
        &self,
        graph: &CapturedGraph,
        group: &FusedKernel,
        saved: &SavedTensorSet,
    ) -> Result<Self::Kernel> {
        let chain = crate::codegen::fuser::chain_from_graph(graph, group, saved)?;
        let fused = chain.fused()?;
        Ok(KernelDefinition::new(
            fused.op_name,
            fused.operand_count,
            fused.dtype,
            fused.expr,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen::{CpuJitKernel, binary_forward, unary_forward};
    use incin_core::compiled::FusionPass;
    use incin_core::graph::{Graph, ValueId};
    use incin_core::prelude::{DTypeId, OperationKind};
    use std::collections::BTreeMap;

    /// Six samples per tensor: includes the `relu` kink at `0.0`, where both
    /// executions must apply the same symbolic derivative convention.
    const XS: [f32; 6] = [1.0, -2.0, 0.5, -0.25, 0.0, 3.0];
    const CS: [f32; 6] = [2.0, 3.0, -1.0, 0.75, 4.0, 0.5];
    /// Non-uniform incoming gradient, so a fusion that dropped a multiply
    /// (or one path through the chain) cannot hide behind ones.
    const GRAD: [f32; 6] = [1.0, 0.5, 2.0, -1.0, 0.25, -0.75];
    const TOL: f32 = 1e-6;

    fn assert_close(got: f32, want: f32, what: &str, index: usize) {
        assert!(
            (got - want).abs() <= TOL,
            "{what}[{index}]: fused {got} vs unfused {want}"
        );
    }

    /// `relu(x) → y; mul(y, c) → out` — the two-node workhorse.
    fn relu_mul_graph() -> (CapturedGraph, ValueId) {
        let mut graph = Graph::new();
        let x = graph.add_value(vec![6], DTypeId::F32, Some("x".into()));
        let c = graph.add_value(vec![6], DTypeId::F32, Some("c".into()));
        let y = graph.add_value(vec![6], DTypeId::F32, Some("y".into()));
        let out = graph.add_value(vec![6], DTypeId::F32, Some("out".into()));
        graph.mark_input(x);
        graph.mark_input(c);
        graph.mark_output(out);
        graph.add_node(OperationKind::Relu, vec![x], vec![y], BTreeMap::new());
        graph.add_node(OperationKind::Mul, vec![y, c], vec![out], BTreeMap::new());
        let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
        (captured, y)
    }

    /// `relu(x) → y; add(y, c) → z; mul(z, x) → out` — a three-node chain
    /// whose final step re-reads the chain's first boundary, and whose
    /// backward wrt `x` has two paths that both have to survive fusion.
    fn relu_add_mul_graph() -> CapturedGraph {
        let mut graph = Graph::new();
        let x = graph.add_value(vec![6], DTypeId::F32, Some("x".into()));
        let c = graph.add_value(vec![6], DTypeId::F32, Some("c".into()));
        let y = graph.add_value(vec![6], DTypeId::F32, Some("y".into()));
        let z = graph.add_value(vec![6], DTypeId::F32, Some("z".into()));
        let out = graph.add_value(vec![6], DTypeId::F32, Some("out".into()));
        graph.mark_input(x);
        graph.mark_input(c);
        graph.mark_output(out);
        graph.add_node(OperationKind::Relu, vec![x], vec![y], BTreeMap::new());
        graph.add_node(OperationKind::Add, vec![y, c], vec![z], BTreeMap::new());
        graph.add_node(OperationKind::Mul, vec![z, x], vec![out], BTreeMap::new());
        CapturedGraph::capture(&graph).expect("capture should succeed")
    }

    /// One `KernelDefinition` per original node — the unfused execution.
    fn stepwise_kernels() -> (CpuJitKernel, CpuJitKernel, CpuJitKernel) {
        let relu = CpuJitKernel::new(KernelDefinition::new(
            "relu",
            1,
            DTypeId::F32,
            unary_forward("relu").expect("relu is in the catalog"),
        ));
        let add = CpuJitKernel::new(KernelDefinition::new(
            "add",
            2,
            DTypeId::F32,
            binary_forward("add").expect("add is in the catalog"),
        ));
        let mul = CpuJitKernel::new(KernelDefinition::new(
            "mul",
            2,
            DTypeId::F32,
            binary_forward("mul").expect("mul is in the catalog"),
        ));
        (relu, add, mul)
    }

    /// Admission through core's public surface, then one lowering call.
    fn admit_and_lower(graph: &CapturedGraph) -> KernelDefinition {
        let candidates = FusionPass.find_candidates(graph);
        let (graph, planned) = FusionPass
            .apply(graph, &candidates, &SavedTensorSet::new())
            .expect("admission is infallible");
        assert_eq!(planned.groups.len(), 1, "refused: {:?}", planned.refused);
        assert!(planned.refused.is_empty(), "{:?}", planned.refused);
        PointwiseScalarLowering
            .lower_group(&graph, &planned.groups[0], &SavedTensorSet::new())
            .expect("proven group should lower")
    }

    /// One admitted two-node group lowers to one kernel whose outputs and
    /// gradients equal the two-launch unfused execution, element for element.
    #[test]
    fn lower_group_emits_one_kernel_from_an_admitted_two_node_group() {
        let (graph, _y) = relu_mul_graph();
        let definition = admit_and_lower(&graph);
        assert_eq!(definition.name, "fused_relu_mul");
        assert_eq!(definition.input_arity, 2);
        assert_eq!(definition.dtype, DTypeId::F32);

        let fused = CpuJitKernel::new(definition);
        let (relu, _add, mul) = stepwise_kernels();

        let mut fused_out = [0.0f32; 6];
        fused
            .eval_f32(&[&XS, &CS], &mut fused_out)
            .expect("fused forward");
        let mut relu_out = [0.0f32; 6];
        let mut unfused_out = [0.0f32; 6];
        relu.eval_f32(&[&XS], &mut relu_out).expect("relu forward");
        mul.eval_f32(&[&relu_out, &CS], &mut unfused_out)
            .expect("mul forward");
        for i in 0..XS.len() {
            assert_close(fused_out[i], unfused_out[i], "forward", i);
        }

        // d/dx: fused symbolic diff vs the two-launch chain rule.
        let mut grad_x_fused = [0.0f32; 6];
        fused
            .eval_backward_f32(&GRAD, &[&XS, &CS], 0, &mut grad_x_fused)
            .expect("fused backward");
        let mut grad_relu = [0.0f32; 6];
        let mut grad_x_unfused = [0.0f32; 6];
        mul.eval_backward_f32(&GRAD, &[&relu_out, &CS], 0, &mut grad_relu)
            .expect("mul backward");
        relu.eval_backward_f32(&grad_relu, &[&XS], 0, &mut grad_x_unfused)
            .expect("relu backward");
        for i in 0..XS.len() {
            assert_close(grad_x_fused[i], grad_x_unfused[i], "grad_x", i);
        }

        // d/dc: `c` feeds the same multiply in both executions.
        let mut grad_c_fused = [0.0f32; 6];
        fused
            .eval_backward_f32(&GRAD, &[&XS, &CS], 1, &mut grad_c_fused)
            .expect("fused grad_c");
        let mut grad_c_unfused = [0.0f32; 6];
        mul.eval_backward_f32(&GRAD, &[&relu_out, &CS], 1, &mut grad_c_unfused)
            .expect("mul grad_c");
        for i in 0..XS.len() {
            assert_close(grad_c_fused[i], grad_c_unfused[i], "grad_c", i);
        }
    }

    /// A three-node admitted group matches the stepwise execution in both
    /// directions: forward outputs, `d/dx` (two paths), and `d/dc`.
    #[test]
    fn a_three_node_admitted_group_matches_stepwise_outputs_and_gradients() {
        let graph = relu_add_mul_graph();
        let definition = admit_and_lower(&graph);
        assert_eq!(definition.name, "fused_relu_add_mul");
        assert_eq!(definition.input_arity, 2);

        let fused = CpuJitKernel::new(definition);
        let (relu, add, mul) = stepwise_kernels();

        let mut fused_out = [0.0f32; 6];
        fused
            .eval_f32(&[&XS, &CS], &mut fused_out)
            .expect("fused forward");
        let mut relu_out = [0.0f32; 6];
        let mut add_out = [0.0f32; 6];
        let mut unfused_out = [0.0f32; 6];
        relu.eval_f32(&[&XS], &mut relu_out).expect("relu forward");
        add.eval_f32(&[&relu_out, &CS], &mut add_out)
            .expect("add forward");
        mul.eval_f32(&[&add_out, &XS], &mut unfused_out)
            .expect("mul forward");
        for i in 0..XS.len() {
            assert_close(fused_out[i], unfused_out[i], "forward", i);
        }

        // d/dx through the chain plus the direct mul operand — both paths
        // must survive the fusion.
        let mut grad_x_fused = [0.0f32; 6];
        fused
            .eval_backward_f32(&GRAD, &[&XS, &CS], 0, &mut grad_x_fused)
            .expect("fused backward");
        let mut grad_add = [0.0f32; 6];
        mul.eval_backward_f32(&GRAD, &[&add_out, &XS], 0, &mut grad_add)
            .expect("mul d/d(add_out)");
        let mut grad_relu = [0.0f32; 6];
        add.eval_backward_f32(&grad_add, &[&relu_out, &CS], 0, &mut grad_relu)
            .expect("add d/d(relu_out)");
        let mut grad_x_chain = [0.0f32; 6];
        relu.eval_backward_f32(&grad_relu, &[&XS], 0, &mut grad_x_chain)
            .expect("relu backward");
        let mut grad_x_direct = [0.0f32; 6];
        mul.eval_backward_f32(&GRAD, &[&add_out, &XS], 1, &mut grad_x_direct)
            .expect("mul d/d(x)");
        for i in 0..XS.len() {
            let unfused = grad_x_chain[i] + grad_x_direct[i];
            assert_close(grad_x_fused[i], unfused, "grad_x", i);
        }

        // d/dc flows only through the add.
        let mut grad_c_fused = [0.0f32; 6];
        fused
            .eval_backward_f32(&GRAD, &[&XS, &CS], 1, &mut grad_c_fused)
            .expect("fused grad_c");
        let mut grad_c_unfused = [0.0f32; 6];
        add.eval_backward_f32(&grad_add, &[&relu_out, &CS], 1, &mut grad_c_unfused)
            .expect("add d/d(c)");
        for i in 0..XS.len() {
            assert_close(grad_c_fused[i], grad_c_unfused[i], "grad_c", i);
        }
    }

    /// Groups that bypassed admission still fail closed at the seam: a
    /// saved intermediate and a three-boundary chain are refused by name.
    #[test]
    fn the_seam_refuses_groups_that_bypass_admission() {
        let (graph, y) = relu_mul_graph();
        let group = FusedKernel {
            source_node_indices: vec![0, 1],
            primary_op: OperationKind::Relu,
        };

        let mut saved = SavedTensorSet::new();
        saved.save(y);
        let err = PointwiseScalarLowering
            .lower_group(&graph, &group, &saved)
            .expect_err("saved intermediate cannot disappear");
        assert!(err.to_string().contains("saved for backward"), "{err}");

        // `out = (x + c1) * c2` needs three boundaries; templates hold two.
        let mut wide = Graph::new();
        let x = wide.add_value(vec![4], DTypeId::F32, Some("x".into()));
        let c1 = wide.add_value(vec![4], DTypeId::F32, Some("c1".into()));
        let c2 = wide.add_value(vec![4], DTypeId::F32, Some("c2".into()));
        let t = wide.add_value(vec![4], DTypeId::F32, Some("t".into()));
        let out = wide.add_value(vec![4], DTypeId::F32, Some("out".into()));
        wide.mark_input(x);
        wide.mark_input(c1);
        wide.mark_input(c2);
        wide.mark_output(out);
        wide.add_node(OperationKind::Add, vec![x, c1], vec![t], BTreeMap::new());
        wide.add_node(OperationKind::Mul, vec![t, c2], vec![out], BTreeMap::new());
        let wide = CapturedGraph::capture(&wide).expect("capture should succeed");
        let wide_group = FusedKernel {
            source_node_indices: vec![0, 1],
            primary_op: OperationKind::Add,
        };
        let err = PointwiseScalarLowering
            .lower_group(&wide, &wide_group, &SavedTensorSet::new())
            .expect_err("three boundaries exceed the template budget");
        assert!(err.to_string().contains("one or two"), "{err}");
    }
}
