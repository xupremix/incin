//! Shared dtype, layout and math-mode vocabulary the capability tables and
//! rule builders draw from.
//!
//! Each constant is the exact set one or more backends can honestly claim
//! for a group of operations; see each constant's own doc comment for why it
//! is not wider or narrower than it is.

use incin_core::exec::{LayoutClass, MathMode};
use incin_core::tensor::dtype::{DTypeDescriptor, DTypeId};

pub(super) const ALL_DTYPES: &[DTypeDescriptor] = &[
    DTypeId::U8.descriptor(),
    DTypeId::U32.descriptor(),
    DTypeId::I64.descriptor(),
    DTypeId::BF16.descriptor(),
    DTypeId::F16.descriptor(),
    DTypeId::F32.descriptor(),
    DTypeId::F64.descriptor(),
    DTypeId::Q8_0.descriptor(),
    DTypeId::Bool.descriptor(),
];
pub(super) const FLOAT_DTYPES: &[DTypeDescriptor] = &[
    DTypeId::BF16.descriptor(),
    DTypeId::F16.descriptor(),
    DTypeId::F32.descriptor(),
    DTypeId::F64.descriptor(),
];
pub(super) const CUDA_STORAGE_DTYPES: &[DTypeDescriptor] = &[
    DTypeId::I64.descriptor(),
    DTypeId::BF16.descriptor(),
    DTypeId::F16.descriptor(),
    DTypeId::F32.descriptor(),
    DTypeId::F64.descriptor(),
];
/// `CUDA_STORAGE_DTYPES` plus `bool`, for the specific CUDA rows verified
/// safe for a 1-byte dtype: `storage` (allocation/`to_bytes`/`from_bytes`)
/// and `reshape` (metadata-only, no kernel launch either way) never assume a
/// fixed element width the way `broadcast_as`'s `shape_op` kernel does.
/// Deliberately a separate constant rather than widening
/// `CUDA_STORAGE_DTYPES` itself: Metal's own `broadcast` row still reuses
/// that constant and has the identical `shape_op`-style byte-width
/// limitation, unverified and out of this session's stated scope — widening
/// the shared constant in place would have silently widened Metal's
/// unverified claim too.
pub(super) const CUDA_BOOL_SAFE_STORAGE_DTYPES: &[DTypeDescriptor] = &[
    DTypeId::I64.descriptor(),
    DTypeId::BF16.descriptor(),
    DTypeId::F16.descriptor(),
    DTypeId::F32.descriptor(),
    DTypeId::F64.descriptor(),
    DTypeId::Bool.descriptor(),
];
pub(super) const F32_ONLY: &[DTypeDescriptor] = &[DTypeId::F32.descriptor()];
/// The union of `where_cond`'s/`masked_fill`'s value dtype and their `bool`
/// mask operand's, for the same reason `INDEX_AND_F32_DTYPES` exists:
/// `dispatch::execute` (`crates/incin-core/src/exec/dispatch.rs`'s
/// `admit_invocation`) checks *every* operand's dtype against the one
/// resolved capability row in turn, so a row narrower than the union of what
/// every operand actually carries makes the operation unreachable — the
/// `mask` operand would fail dtype admission before either kernel ever
/// launches. Not a claim that either operand may be *either* dtype: the
/// descriptor's own per-operand contract (`exec/catalog`'s
/// `WhereCond`/`MaskedFill` arms) and `cuda::ops::select`'s own
/// `require_bool_mask`/`cuda_require_f32` checks enforce the real, tighter
/// per-operand split this row cannot state on its own.
pub(super) const F32_AND_BOOL: &[DTypeDescriptor] =
    &[DTypeId::F32.descriptor(), DTypeId::Bool.descriptor()];
/// `logical_and`/`logical_or`/`logical_not`: every operand and the output
/// are `bool`, unlike `where_cond`/`masked_fill`'s mixed `F32_AND_BOOL`, so
/// one dtype suffices — no union needed.
pub(super) const BOOL_ONLY: &[DTypeDescriptor] = &[DTypeId::Bool.descriptor()];
/// The only quantized representation any backend implements today.
pub(super) const Q8_ONLY: &[DTypeDescriptor] = &[DTypeId::Q8_0.descriptor()];
pub(super) const NON_QUANTIZED: &[DTypeDescriptor] = &[
    DTypeId::U8.descriptor(),
    DTypeId::U32.descriptor(),
    DTypeId::I64.descriptor(),
    DTypeId::BF16.descriptor(),
    DTypeId::F16.descriptor(),
    DTypeId::F32.descriptor(),
    DTypeId::F64.descriptor(),
    DTypeId::Bool.descriptor(),
];
/// The union of an integer index operand's dtypes and an f32 data operand's.
///
/// Two operations have exactly this shape: `embedding` (integer indices, f32
/// weight table) and `cross_entropy_loss` (f32 logits, integer class
/// targets). Not a claim that either operand may be *either* — the
/// descriptor's own per-operand contract and `cpu::canonical`'s `f32_only`
/// both enforce the real, tighter split this row cannot state on its own,
/// because `dispatch::execute` applies one dtype set to every operand in
/// turn. See the `embedding` and `composed_reduction_indexed` groups' own
/// comments in `cpu_descriptor_operations!`.
pub(super) const INDEX_AND_F32_DTYPES: &[DTypeDescriptor] = &[
    DTypeId::U8.descriptor(),
    DTypeId::U32.descriptor(),
    DTypeId::I64.descriptor(),
    DTypeId::F32.descriptor(),
];
pub(super) const CONTIGUOUS: &[LayoutClass] = &[LayoutClass::Contiguous];
pub(super) const CPU_LAYOUTS: &[LayoutClass] = &[LayoutClass::Contiguous, LayoutClass::Strided];
pub(super) const PRECISE: &[MathMode] = &[MathMode::Precise];
