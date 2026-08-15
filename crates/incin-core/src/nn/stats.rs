//! Runtime model statistics (parameter count, multiply-accumulates) — the v1
//! ("ships now, not `const`") half of `docs/growth/04-compile-time-stats.md`.
//! `ComputeStats` is auto-derived for every `#[module]` struct (summing each
//! field's contribution, exactly like `NamedLayers` and state traversal; a
//! leaf layer with a known compute formula (currently just [`crate::nn::linear::Linear`])
//! opts out of that default via `#[module(no_stats)]` and hand-implements
//! this trait with its real formula instead.
//!
//! **Scope of this v1, stated plainly so it can't be quietly overclaimed:**
//! parameter counts are exact for every model (they're just element counts,
//! independent of batch size or spatial input size). MAC counts are exact
//! for `Linear` (its formula only needs its own weight shape) but are **not
//! yet computed for `Conv1d`/`Conv2d`** (0, not "unknown") — their MACs
//! formula additionally needs the input's spatial size, which isn't part of
//! a conv layer's own stored state and isn't available without either a real
//! forward pass or v2's type-level shape propagation. Both are out of scope
//! here; this is an honest gap, not a bug.

use crate::nn::param::{Buffer, Param, TrainState};
use crate::nn::{AdaptiveAvgPool2d, GELU, ReLU, Sequential, Sigmoid, Softmax, Swish, Tanh};
use crate::shapes::error::OperationKind;
use crate::shapes::{DynShape, Shape, ShapeBuf};
use crate::tensor::backend::Backend;
use crate::tensor::dtype::DType;
use alloc::vec::Vec;

/// One layer's (or one subtree's, once summed) contribution to a model's
/// stats: parameter count and multiply-accumulate count for one forward
/// pass at a given batch size.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LayerStats {
    /// Number of trainable parameter elements (buffers — e.g. BatchNorm's
    /// running mean/var — are deliberately excluded, matching
    /// the typed `VisitParameters` traversal's trainable-only convention).
    pub params: u64,
    /// Multiply-accumulate operations for one forward pass at the batch
    /// size `compute_stats` was called with. 0 for layers with no known
    /// formula yet (see module docs) — not the same as "no compute at all."
    pub macs: u64,
}

impl core::ops::Add for LayerStats {
    type Output = LayerStats;
    fn add(self, rhs: Self) -> Self::Output {
        LayerStats {
            params: self.params + rhs.params,
            macs: self.macs + rhs.macs,
        }
    }
}

impl core::ops::AddAssign for LayerStats {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl core::iter::Sum for LayerStats {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), core::ops::Add::add)
    }
}

/// A model's total stats, in the units a budget check or a printed report
/// actually wants: `flops` is the conventional `2 * macs` (one multiply +
/// one add per MAC), reported alongside the raw MAC count since both
/// conventions show up in the wild.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModelStats {
    /// Total trainable parameter elements.
    pub params: u64,
    /// Total multiply-accumulate operations for one forward pass.
    pub macs: u64,
    /// `2 * macs` — the conventional FLOPs figure.
    pub flops: u64,
}

/// Implemented once per compute-bearing leaf layer with a known formula
/// (opts out of the `#[module]`-derived default via `#[module(no_stats)]`),
/// and auto-derived for every other `#[module]` struct as "sum of my
/// fields' stats." Container types (`Sequential`, `Option`) and the raw
/// [`crate::nn::param::Param`] leaf are hand-implemented below, mirroring
/// how `NamedLayers` handles the same cases.
pub trait ComputeStats {
    /// This layer's (or, for a composite type, this whole subtree's)
    /// parameter count and MAC count for one forward pass at `batch`.
    fn compute_stats(&self, batch: u64) -> LayerStats;

    /// Convenience entry point: `model.stats(batch)` instead of
    /// `model.compute_stats(batch).into_model_stats()`. Not overridable —
    /// every implementor gets this for free.
    fn stats(&self, batch: u64) -> ModelStats {
        self.compute_stats(batch).into_model_stats()
    }
}

impl<
    S: Shape + DynShape,
    B: crate::tensor::backend::VariableBackend,
    K: DType,
    Train: crate::nn::param::TrainState,
> ComputeStats for Param<S, B, K, Train>
{
    /// A parameter's own element count; it has no operation of its own, so
    /// 0 MACs (the layer that *uses* this parameter reports those, if it
    /// has a known formula).
    fn compute_stats(&self, _batch: u64) -> LayerStats {
        LayerStats {
            params: validated_parameter_count(&self.shape_dims()),
            macs: 0,
        }
    }
}

impl<S: Shape + DynShape, B: crate::tensor::backend::VariableBackend, K: DType> ComputeStats
    for Buffer<S, B, K>
{
    fn compute_stats(&self, _batch: u64) -> LayerStats {
        LayerStats::default()
    }
}

pub(crate) fn validated_parameter_count(dims: &[usize]) -> u64 {
    let elements = ShapeBuf::from_slice(dims)
        .checked_numel(OperationKind::Storage)
        .expect("parameter tensor shape crossed checked construction");
    u64::try_from(elements).expect("usize parameter count must fit u64")
}

impl<T: ComputeStats> ComputeStats for Option<T> {
    /// Delegates to the wrapped value's stats; nothing for `None`.
    fn compute_stats(&self, batch: u64) -> LayerStats {
        match self {
            Some(v) => v.compute_stats(batch),
            None => LayerStats::default(),
        }
    }
}

impl<L1: ComputeStats, L2: ComputeStats> ComputeStats for Sequential<L1, L2> {
    /// Sum of both halves' stats.
    fn compute_stats(&self, batch: u64) -> LayerStats {
        self.0.compute_stats(batch) + self.1.compute_stats(batch)
    }
}

macro_rules! impl_zero_compute_stats {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl ComputeStats for $ty {
                /// No parameters, no MACs.
                fn compute_stats(&self, _batch: u64) -> LayerStats {
                    LayerStats::default()
                }
            }
        )+
    };
}
impl_zero_compute_stats!(ReLU, GELU, Swish, Sigmoid, Tanh, Softmax);

/// Aggregates a slice of already-computed per-field [`LayerStats`] — a tiny
/// helper so `#[module]`'s generated code has one place to sum, matching
/// the style of [`crate::nn::module::format_layer_summary`]'s helpers.
pub fn sum_stats(items: &[LayerStats]) -> LayerStats {
    items.iter().copied().sum()
}

impl LayerStats {
    /// Wraps this subtree's stats as a top-level [`ModelStats`] report
    /// (`flops = 2 * macs`, the conventional multiply+add-per-MAC count).
    pub fn into_model_stats(self) -> ModelStats {
        ModelStats {
            params: self.params,
            macs: self.macs,
            flops: self.macs * 2,
        }
    }
}

#[cfg(test)]
mod tests {
    // `s!` is written for consumers of the `incin` facade, so it expands to
    // `::incin::prelude::…`. `s![@ ..]` is the in-crate form and expands to
    // the owning modules directly — which is what this module needs, since
    // `incin-core` does not depend on `incin`. The integration crates under
    // `tests/` use `extern crate incin_core as incin;`, which does create the
    // crate-root entry the absolute path resolves against; a `use crate as
    // incin` alias, which is what stood here before `CI-005`, does not.
    use super::*;
    use crate::backend_authoring::{Backend, VariableBackend};
    use crate::nn::{Linear, NamedLayers, ReLU, Sequential};
    use crate::tensor::device::Cpu;
    use incin_macros::{module, s};

    #[test]
    fn layer_stats_add_sums_both_fields() {
        let a = LayerStats {
            params: 10,
            macs: 20,
        };
        let b = LayerStats { params: 3, macs: 4 };
        assert_eq!(
            a + b,
            LayerStats {
                params: 13,
                macs: 24
            }
        );
    }

    #[test]
    fn layer_stats_sum_over_empty_iter_is_default() {
        let items: Vec<LayerStats> = Vec::new();
        assert_eq!(sum_stats(&items), LayerStats::default());
    }

    #[test]
    fn into_model_stats_doubles_macs_for_flops() {
        let stats = LayerStats {
            params: 5,
            macs: 100,
        }
        .into_model_stats();
        assert_eq!(
            stats,
            ModelStats {
                params: 5,
                macs: 100,
                flops: 200
            }
        );
    }

    // Same shape as `docs/growth/04-compile-time-stats.md`'s own worked
    // example (784 -> 128 -> 10), which independently states the expected
    // totals (101,770 params / 101,632 MACs at batch=1) — matching those
    // exactly here is doubly-confirming: both this implementation and the
    // doc's hand math agree.
    #[module(internal)]
    struct TestMlp<Bk: Backend + VariableBackend> {
        fc1: Linear<s![@ 784, 128], Bk>,
        fc2: Linear<s![@ 128, 10], Bk>,
    }

    type TestBackend = crate::test_utils::DummyBackend<Cpu>;

    fn build_test_mlp() -> TestMlp<TestBackend> {
        TestMlp {
            fc1: Linear::build(()).unwrap(),
            fc2: Linear::build(()).unwrap(),
        }
    }

    #[test]
    fn mlp_stats_match_the_growth_doc_hand_computed_numbers_at_batch_1() {
        let model = build_test_mlp();
        let stats = model.stats(1);
        // fc1: 784*128 + 128 = 100,480 params; 784*128*1 = 100,352 MACs.
        // fc2: 128*10 + 10 = 1,290 params; 128*10*1 = 1,280 MACs.
        assert_eq!(stats.params, 101_770);
        assert_eq!(stats.macs, 101_632);
        assert_eq!(stats.flops, 203_264);
    }

    #[test]
    fn mlp_macs_scale_linearly_with_batch_but_params_do_not() {
        let model = build_test_mlp();
        let stats4 = model.stats(4);
        assert_eq!(stats4.params, 101_770, "params must not depend on batch");
        assert_eq!(stats4.macs, 101_632 * 4);
    }

    #[test]
    fn sequential_of_linear_and_relu_sums_correctly_and_relu_contributes_nothing() {
        let seq = Sequential(Linear::<s![@ 16, 8], TestBackend>::build(()).unwrap(), ReLU);
        let stats = seq.stats(2);
        // Only the Linear side has params/MACs; ReLU is a verified 0/0 no-op.
        assert_eq!(stats.params, 16 * 8 + 8);
        assert_eq!(stats.macs, 16 * 8 * 2);
    }

    #[test]
    fn summary_with_stats_appends_a_readable_totals_footer() {
        let model = build_test_mlp();
        let text = crate::nn::module::format_layer_summary_with_stats(
            &model.layer_structure(""),
            model.stats(1),
        );
        assert!(
            text.contains("Total params: 101770"),
            "missing/wrong params footer in:\n{text}"
        );
        assert!(
            text.contains("MACs: 101632"),
            "missing/wrong MACs footer in:\n{text}"
        );
        assert!(
            text.contains("FLOPs: 203264"),
            "missing/wrong FLOPs footer in:\n{text}"
        );
    }

    #[test]
    fn named_layers_summary_with_stats_convenience_method_matches_the_free_function() {
        let model = build_test_mlp();
        assert_eq!(
            model.summary_with_stats(1),
            crate::nn::module::format_layer_summary_with_stats(
                &model.layer_structure(""),
                model.stats(1)
            )
        );
    }
}
