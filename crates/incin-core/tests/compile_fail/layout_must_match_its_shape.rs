//! A layout cannot be attached to a tensor of a different shape.
//!
//! This exact type compiled before the layout parameter moved onto the trait: a
//! rank-two tensor carrying a rank-three row-major claim. `Tensor` bounded its
//! layout on a shape-free `Layout`, and the congruence trait that would have
//! rejected it -- `LayoutOf<S>` -- was bounded nowhere in the crate, appearing
//! in two tests and no production code. The claim and the shape were free to
//! disagree.
//!
//! `Layout<S>` closes it by construction rather than by a check: the marker has
//! no shape of its own to be wrong about, so the mismatch cannot be spelled.
//! What fails here is naming `RowMajor` with an argument at all.
//!
//! The rank half is pinned separately by
//! `channels_last_needs_a_rank_four_shape.rs`.
extern crate incin_core as incin;
use incin_backends::cpu::CpuBackendImpl;
use incin_core::prelude::*;
use incin_core::shapes::RowMajor;
use incin_macros::s;

type Mismatched = Tensor<s![2, 3], CpuBackendImpl, f32, NoGrad, Local, RowMajor<s![9, 9, 9]>>;

fn takes(_: Mismatched) {}

fn main() {
    let _ = takes;
}
