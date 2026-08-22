//! Runtime model statistics (parameter count, multiply-accumulates) - the v1
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
//! yet computed for `Conv1d`/`Conv2d`** (0, not "unknown") - their MACs
//! formula additionally needs the input's spatial size, which isn't part of
//! a conv layer's own stored state and isn't available without either a real
//! forward pass or v2's type-level shape propagation. Both are out of scope
//! here; this is an honest gap, not a bug.

use crate::nn::param::{Buffer, Param};
use crate::nn::{Dropout, ELU, GELU, Mish, ReLU, Sequential, Sigmoid, Softmax, Swish, Tanh};
use crate::shapes::error::OperationKind;
use crate::shapes::{DynShape, Shape, ShapeBuf};
use crate::tensor::dtype::DType;

/// One layer's (or one subtree's, once summed) contribution to a model's
/// stats: parameter count and multiply-accumulate count for one forward
/// pass at a given batch size.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct LayerStats {
    /// Number of trainable parameter elements (buffers - e.g. BatchNorm's
    /// running mean/var - are deliberately excluded, matching
    /// the typed `VisitParameters` traversal's trainable-only convention).
    pub params: u64,
    /// Multiply-accumulate operations for one forward pass at the batch
    /// size `compute_stats` was called with. 0 for layers with no known
    /// formula yet (see module docs) - not the same as "no compute at all."
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
    /// `2 * macs` - the conventional FLOPs figure.
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
    /// `model.compute_stats(batch).into_model_stats()`. Not overridable -
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
impl_zero_compute_stats!(
    ReLU, GELU, Swish, Mish, ELU, Sigmoid, Tanh, Softmax, Dropout
);

impl<K: typenum::Unsigned, S: typenum::Unsigned, P: typenum::Unsigned, D: typenum::Unsigned>
    ComputeStats for crate::nn::max_pool2d::MaxPool2d<K, S, P, D>
{
    fn compute_stats(&self, _batch: u64) -> LayerStats {
        LayerStats::default()
    }
}

impl<K: typenum::Unsigned, S: typenum::Unsigned, P: typenum::Unsigned, D: typenum::Unsigned>
    ComputeStats for crate::nn::avg_pool2d::AvgPool2d<K, S, P, D>
{
    fn compute_stats(&self, _batch: u64) -> LayerStats {
        LayerStats::default()
    }
}

impl<HOut: typenum::Unsigned, WOut: typenum::Unsigned> ComputeStats
    for crate::nn::adaptive_avg_pool2d::AdaptiveAvgPool2d<HOut, WOut>
{
    fn compute_stats(&self, _batch: u64) -> LayerStats {
        LayerStats::default()
    }
}

/// Aggregates a slice of already-computed per-field [`LayerStats`] - a tiny
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

// Everything below needs only the arithmetic of `LayerStats` itself, so it
// stays an in-crate unit test. The layer-level and model-level numbers need a
// real backend to build `Linear` against, and a backend crate cannot be linked
// into this crate's own unit tests without duplicating `incin-core`; those
// tests live in `tests/model_stats.rs`.
#[cfg(test)]
mod tests {
    use super::*;

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
}
