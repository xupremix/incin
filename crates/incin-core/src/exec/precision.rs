//! Floating-point precision policies and loss scaling for mixed-precision training.

use crate::prelude::DTypeId;

/// Loss scaling policy for mixed-precision numerical stability.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum LossScaling {
    /// No loss scaling applied.
    #[default]
    None,
    /// Static loss scaling with a fixed multiplier.
    Static(f32),
    /// Dynamic loss scaling that adjusts based on non-finite gradient detection.
    Dynamic {
        /// Current scale factor.
        scale: f32,
        /// Factor by which to multiply scale when no non-finite gradients occur.
        growth_factor: f32,
        /// Factor by which to multiply scale when a non-finite gradient occurs.
        backoff_factor: f32,
        /// Number of consecutive finite steps required before growing scale.
        growth_interval: usize,
        /// Consecutive finite steps recorded since the last overflow.
        steps_since_last_overflow: usize,
    },
}

impl LossScaling {
    /// Creates a static loss scaling policy.
    #[must_use]
    pub const fn static_scale(scale: f32) -> Self {
        Self::Static(scale)
    }

    /// Creates a dynamic loss scaling policy with recommended defaults.
    #[must_use]
    pub const fn dynamic_default() -> Self {
        Self::Dynamic {
            scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            steps_since_last_overflow: 0,
        }
    }

    /// Creates a custom dynamic loss scaling policy.
    #[must_use]
    pub const fn dynamic(
        initial_scale: f32,
        growth_factor: f32,
        backoff_factor: f32,
        growth_interval: usize,
    ) -> Self {
        Self::Dynamic {
            scale: initial_scale,
            growth_factor,
            backoff_factor,
            growth_interval,
            steps_since_last_overflow: 0,
        }
    }

    /// Returns the current scale factor.
    #[must_use]
    pub fn scale(&self) -> f32 {
        match *self {
            Self::None => 1.0,
            Self::Static(scale) => scale,
            Self::Dynamic { scale, .. } => scale,
        }
    }

    /// Updates dynamic loss scaling based on whether non-finite (NaN/Inf) values were found.
    pub fn update(&mut self, found_nan_or_inf: bool) {
        if let Self::Dynamic {
            scale,
            growth_factor,
            backoff_factor,
            growth_interval,
            steps_since_last_overflow,
        } = self
        {
            if found_nan_or_inf {
                *scale = (*scale * *backoff_factor).max(1.0);
                *steps_since_last_overflow = 0;
            } else {
                *steps_since_last_overflow += 1;
                if *steps_since_last_overflow >= *growth_interval {
                    *scale *= *growth_factor;
                    *steps_since_last_overflow = 0;
                }
            }
        }
    }
}

impl core::hash::Hash for LossScaling {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match *self {
            Self::None => {}
            Self::Static(s) => s.to_bits().hash(state),
            Self::Dynamic {
                scale,
                growth_factor,
                backoff_factor,
                growth_interval,
                steps_since_last_overflow,
            } => {
                scale.to_bits().hash(state);
                growth_factor.to_bits().hash(state);
                backoff_factor.to_bits().hash(state);
                growth_interval.hash(state);
                steps_since_last_overflow.hash(state);
            }
        }
    }
}

impl Eq for LossScaling {}

/// Precision policy specifying datatypes for storage, compute, accumulation, and output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrecisionPolicy {
    /// Datatype used for stored parameters.
    pub parameter: DTypeId,
    /// Datatype used for operation computation.
    pub compute: DTypeId,
    /// Datatype used for reduction/normalization accumulators.
    pub accumulator: DTypeId,
    /// Datatype produced for tensor outputs.
    pub output: DTypeId,
    /// Loss scaling strategy.
    pub loss_scaling: LossScaling,
}

impl Default for PrecisionPolicy {
    fn default() -> Self {
        Self::fp32()
    }
}

impl PrecisionPolicy {
    /// Standard full-precision float32 policy.
    #[must_use]
    pub const fn fp32() -> Self {
        Self {
            parameter: DTypeId::F32,
            compute: DTypeId::F32,
            accumulator: DTypeId::F32,
            output: DTypeId::F32,
            loss_scaling: LossScaling::None,
        }
    }

    /// Mixed-precision float16 policy with dynamic loss scaling.
    #[must_use]
    pub const fn mixed_f16() -> Self {
        Self {
            parameter: DTypeId::F32,
            compute: DTypeId::F16,
            accumulator: DTypeId::F32,
            output: DTypeId::F16,
            loss_scaling: LossScaling::dynamic_default(),
        }
    }

    /// Mixed-precision bfloat16 policy.
    #[must_use]
    pub const fn mixed_bf16() -> Self {
        Self {
            parameter: DTypeId::F32,
            compute: DTypeId::BF16,
            accumulator: DTypeId::F32,
            output: DTypeId::BF16,
            loss_scaling: LossScaling::None,
        }
    }

    /// Custom precision policy.
    #[must_use]
    pub const fn custom(
        parameter: DTypeId,
        compute: DTypeId,
        accumulator: DTypeId,
        output: DTypeId,
        loss_scaling: LossScaling,
    ) -> Self {
        Self {
            parameter,
            compute,
            accumulator,
            output,
            loss_scaling,
        }
    }

    /// Sets loss scaling for this policy.
    #[must_use]
    pub const fn with_loss_scaling(mut self, loss_scaling: LossScaling) -> Self {
        self.loss_scaling = loss_scaling;
        self
    }
}
