//! Generic parameter optimization: update-rule traits, gradient clipping,
//! and the concrete optimizers built on them.
//!
//! Split by concern per `docs/CONVENTIONS.md`: `traits` is `Optimizer`/
//! `ScaledOptimizer` (the update-rule vocabulary) and `OptimizerBackend`/
//! `ValueClippingBackend` (the canonical execution profile a backend
//! implements to support them, plus the blanket impl over exact-op
//! descriptors that satisfies it); `group` is `ParameterGroup` and the two
//! private types its constructor and every optimizer's update loop build
//! on; `support` is the validation and update-staging logic shared across
//! `step`/`load_state_dict`; `clip` is the public gradient-clipping API;
//! `optimizers` is `SGD`, `AdamW` and `Adam` themselves.

pub mod scheduler;

mod clip;
mod group;
mod optimizers;
mod support;
mod traits;

pub use crate::autograd::Gradients;
pub use clip::{clip_grad_norm, clip_grad_value};
pub use group::ParameterGroup;
pub use optimizers::{Adam, AdamW, SGD};
pub use scheduler::{ConstantLR, LRScheduler, LinearLR};
#[cfg(feature = "std")]
pub use scheduler::{CosineAnnealingLR, StepLR};
pub use traits::{Optimizer, OptimizerBackend, ScaledOptimizer, ValueClippingBackend};
