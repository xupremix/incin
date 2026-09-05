//! `into_shape` re-types a buffer under another shape *for the same dims*:
//! `S2::try_from_dims` validates the tensor's existing dims against `S2` and
//! returns those very dims, so it cannot change an extent. The layout claim has
//! to travel with the retyping, and `RestateFor` is what says a claim can.
//!
//! `ChannelsLast` does not implement it, so this is refused. That refusal is
//! conservative rather than proven: because the dims are preserved, and because
//! channels-last strides are a function of the dims alone, restating
//! `ChannelsLast` as `ChannelsLast` over identical dims would in fact
//! be sound. The impl is missing, not impossible.
//!
//! The real guard against reinterpreting a channels-last buffer under different
//! extents is `Contiguous` on `reshape_view`, which is a genuine reshape and
//! has its own case beside this one.
//!
//! This case exists because the refusal was enforced by nothing but an absent
//! impl, and an absence is invisible to a reader and to a refactor. If the
//! layout traits are reworked -- for instance to drop the shape parameter from
//! the markers -- this is the case that will say whether the behaviour moved.
//!
//! Written against a parameter rather than a constructed value for the same
//! reason as the `reshape_view` case: a channels-last tensor cannot be built
//! yet, so constructing one would fail for a second, unrelated reason.
extern crate incin_core as incin;
use incin_backends::cpu::CpuBackendImpl;
use incin_core::prelude::*;
use incin_core::shapes::ChannelsLast;
use incin_macros::s;

type Nchw = s![1, 2, 2, 2];
type Permuted = s![2, 1, 2, 2];

fn restate(
    t: Tensor<Nchw, CpuBackendImpl, f32, NoGrad, Local, ChannelsLast>,
) -> Tensor<Permuted, CpuBackendImpl, f32, NoGrad, Local, ChannelsLast> {
    t.into_shape::<Permuted>().unwrap()
}

fn main() {
    let _ = restate;
}
