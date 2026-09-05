//! `into_shape` re-describes a buffer under a new shape type without touching
//! it, so the layout claim has to travel with it. `RestateFor` is what states a
//! claim can be re-expressed against another shape, and `ChannelsLast` does not
//! implement it: channels-last strides are a function of the extents, so the
//! same buffer described under different extents is a different, and false,
//! claim. `[1, 2, 2, 2]` is `[8, 1, 4, 2]`; `[2, 1, 2, 2]` is `[4, 1, 2, 1]`.
//!
//! Only `Dyn` and `RowMajor` implement `RestateFor`, and both can honestly
//! restate: `Dyn` claims nothing, and a dense row-major run stays dense under
//! any shape with the same element count.
//!
//! This refusal was enforced by nothing but the absence of an impl, and absence
//! is invisible to a reader and to a refactor. This case makes it a check.
//!
//! Written against a parameter rather than a constructed value for the same
//! reason as the `reshape_view` case beside it: a channels-last tensor cannot
//! be built yet, so constructing one would fail for a second, unrelated reason.
extern crate incin_core as incin;
use incin_backends::cpu::CpuBackendImpl;
use incin_core::prelude::*;
use incin_core::shapes::ChannelsLast;
use incin_macros::s;

type Nchw = s![1, 2, 2, 2];
type Permuted = s![2, 1, 2, 2];

fn restate(
    t: Tensor<Nchw, CpuBackendImpl, f32, NoGrad, Local, ChannelsLast<Nchw>>,
) -> Tensor<Permuted, CpuBackendImpl, f32, NoGrad, Local, ChannelsLast<Permuted>> {
    t.into_shape::<Permuted>().unwrap()
}

fn main() {
    let _ = restate;
}
