//! Concrete CPU pointwise helpers used by descriptor executors and composed
//! backend operations.
//!
//! Every op here resolves the broadcast output shape via
//! `stride::broadcast_shape`, then iterates the OUTPUT shape's logical index
//! space, resolving each operand's own index through its own strides with
//! wraparound (stride-0-equivalent) logic on right-aligned/expanded
//! dimensions — it never pre-materializes a broadcast copy of either operand
//! (the anti-pattern flagged in RESEARCH.md). Every op pushes a `TapeEntry`
//! whose backward closure calls `tape::unbroadcast` on the ORIGINAL
//! (pre-broadcast) operand shapes.
//!
//! Split by concern per `docs/CONVENTIONS.md`: `index` is flat/nd index
//! math shared by several ops; `dispatch` is the generic broadcasting
//! machinery every op below is built from; `unary` is the single-operand
//! `canonical_*` family; `binary` is the two-operand `canonical_*` family
//! plus `add`/`sub`/`mul`/`div_storage`; `softmax` is `canonical_softmax`
//! and `log_softmax`, which need a `Device` type parameter the rest don't.

use incin_core::error::Result;
use incin_core::tensor::device::Device;
use incin_core::tensor::dtype::DType;
use rayon::prelude::*;

use crate::cpu::ops::elementwise_kernel::{self, BinaryOp, UnaryOp};
use crate::cpu::storage::{CpuBuffer, CpuStorage};
use crate::cpu::tape::{self, TapeEntry};
use crate::iteration::{IterationPlan, OperandLayout};

mod binary;
mod dispatch;
mod index;
mod softmax;
#[cfg(test)]
mod tests;
mod unary;

pub(crate) use binary::{
    add_storage, add_storage_with_shape, canonical_atan2, canonical_fmod, canonical_remainder,
    div_storage, div_storage_with_shape, mul_storage, mul_storage_with_shape, sub_storage,
    sub_storage_with_shape,
};
pub(crate) use dispatch::{canonical_unary, elementwise_binary, elementwise_unary};
// Cross-submodule wiring: `unary`/`binary` build every specific op from
// this shared broadcasting/dispatch machinery. None of these four are part
// of the crate's public surface, so each is `pub(super)` in `dispatch` and
// re-exported here with plain (private) `use`, which carries that same
// visibility to `unary.rs`/`binary.rs`'s `use super::*;` without widening it
// further.
use dispatch::{
    canonical_unary_with_deriv_op, elementwise_binary_numeric, elementwise_unary_typed, negate,
};
pub(crate) use index::{flat_to_nd, increment_index};
pub(crate) use softmax::{canonical_softmax, log_softmax};
pub(crate) use unary::{
    canonical_abs, canonical_acos, canonical_acosh, canonical_add_scalar, canonical_asin,
    canonical_asinh, canonical_atan, canonical_atanh, canonical_clamp, canonical_cosh,
    canonical_elu, canonical_erf, canonical_exp, canonical_frac, canonical_gelu, canonical_log,
    canonical_mish, canonical_mul_scalar, canonical_neg, canonical_powf, canonical_relu,
    canonical_rsqrt, canonical_sigmoid, canonical_sinh, canonical_sqrt, canonical_step,
    canonical_swish, canonical_tan, canonical_tanh, canonical_trunc,
};
