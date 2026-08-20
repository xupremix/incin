//! `conv1d`/`conv2d` for CPU storage via im2col (window-unfold into
//! a column matrix) + `ops::matmul::batched_matmul_impl` for the actual
//! multiply-accumulate (D-01).
//!
//! `im2col_1d`/`im2col_2d` are materializing gather loops (NOT a
//! `CpuStorage` view op, per Pitfall 2): every gathered element whose
//! computed source position falls outside the (unpadded) input range is
//! substituted with `0.0` rather than erroring or reading adjacent unrelated
//! buffer memory. `col2im_1d`/`col2im_2d` are the exact inverse scatter-ADD
//! (`+=`, never `=`) fold, so overlapping output windows (stride <
//! kernel_size) correctly accumulate their gradient contributions (Pitfall 5's
//! discipline, applied to conv's own backward).
//!
//! `groups` support (including the depthwise `groups == Cin` degenerate case)
//! is implemented as a single generic `0..groups` loop - narrow the input's
//! channel axis and the weight's output-channel axis into per-group slices,
//! `im2col` + `batched_matmul_impl` + concat - with NO special-casing branch
//! for any particular `groups` value (Pitfall 7).
//!
//! `conv1d_impl`/`conv2d_windowed_impl` each push exactly ONE top-level
//! `TapeEntry`:
//! the im2col unfold/col2im fold steps are plain (non-tape-tracked) helper
//! functions operating on already-materialized `CpuStorage` values, so
//! their OWN backward is hand-composed here (reusing `batched_matmul_impl`'s
//! already-gradcheck-verified backward only for the INTERNAL
//! multiply-accumulate step, per RESEARCH.md Pattern 3). Bias, when present,
//! is broadcast-added via the canonical storage helper AFTER
//! the hand-composed conv math, so `grad_bias` falls out of that op's own
//! existing backward + `unbroadcast` for free - it is never hand-derived
//! inside `conv1d_impl`/`conv2d_windowed_impl`'s own closure.
//!
//! `conv_transpose2d_impl` (Plan 04-07, RESEARCH.md Pattern 4) reuses
//! `col2im_2d` VERBATIM as its own forward fold subroutine - transposed
//! convolution's forward pass is exactly `conv2d`'s own backward-data
//! (grad-w.r.t.-input) formula applied to `input` directly instead of to a
//! gradient. Its own backward, symmetrically, reuses `im2col_2d` +
//! `batched_matmul_impl` (i.e. `conv2d`'s FORWARD formula) to recover
//! `grad_input`. `output_padding` is handled as a separate final
//! allocate-larger-then-copy-into-leading-sub-region step (via
//! `scatter_into_zeros`), never folded into `padding`'s own symmetric
//! offset arithmetic (Pitfall 4). Only `groups == 1` is supported, matching
//! `CandleBackend::conv_transpose2d`'s own confirmed effective behavior.
//!
//! Split by concern per `docs/CONVENTIONS.md`: `helpers` is the shared
//! output-size arithmetic and group validation every variant below builds
//! on; `unfold1d` is `im2col_1d`/`col2im_1d`; `window` is the `Window2d`
//! descriptor plus their 2D generalization `im2col_2d`/`col2im_2d`;
//! `conv1d`, `conv2d`, and `conv_transpose2d` are the three canonical
//! implementations, one per file; `combine` is the plain (non-tape-tracked)
//! backward-composition helpers (`sum_batch_dim`, `concat_along_dim0/1`)
//! their hand-composed backward closures share.

mod combine;
mod conv1d;
mod conv2d;
mod conv_transpose2d;
mod helpers;
mod unfold1d;
mod window;

#[cfg(test)]
mod tests;

pub(crate) use conv_transpose2d::conv_transpose2d_impl;
pub(crate) use conv1d::conv1d_impl;
pub(crate) use conv2d::conv2d_windowed_impl;
pub(crate) use window::Window2d;
