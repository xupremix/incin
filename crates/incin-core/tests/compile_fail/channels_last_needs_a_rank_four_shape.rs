//! Channels-last is defined against NCHW, so it is meaningless at any other
//! rank -- and now unnameable there.
//!
//! `Layout::STATIC_STRIDES` could already tell rank four from anything else,
//! but only at the value level: `ChannelsLast` implemented `Layout<S>` for
//! every `S` and reported an empty stride list when the rank was wrong. That
//! left `Tensor<s![2, 3], .., ChannelsLast>` a nameable type carrying a layout
//! that claimed nothing, because the rank test lived in a constant rather than
//! in a bound.
//!
//! Rank is not directly expressible as a bound -- `Shape::RANK` is an
//! associated `Option<usize>`, and a constant cannot gate an impl -- but it is
//! expressible structurally, which is what `Rank4` does: four `DimCons` cells
//! terminated by `Nil`, or a `Ranked<U4>`. `Dyn` is excluded on purpose, since
//! a runtime rank cannot prove it is four.
extern crate incin_core as incin;
use incin_backends::cpu::CpuBackendImpl;
use incin_core::prelude::*;
use incin_core::shapes::ChannelsLast;
use incin_macros::s;

fn rank_two(t: Tensor<s![2, 3], CpuBackendImpl, f32, NoGrad, Local, ChannelsLast>) {
    let _ = t;
}

fn main() {
    let _ = rank_two;
}
