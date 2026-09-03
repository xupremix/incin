//! A downstream layout cannot claim to describe a fresh allocation.
//!
//! `Tensor::zeros` is generic over the layout parameter so that asking for
//! `Dense<S, B>` yields a real `RowMajor<S>` from the allocation itself. That
//! generality is only safe because its `FreshDense` bound is sealed: a freshly
//! allocated buffer is genuinely row-major, and genuinely nothing else. Without
//! the seal a constructor becomes a minting press -- name any layout and get a
//! tensor claiming it, with no unsafe block and no runtime check anywhere near
//! the lie.
//!
//! `Sideways` below is a perfectly well-formed `Layout`: it implements every
//! trait a layout outside this crate can implement. The seal is what stops it
//! going further.
//!
//! # Why this pins the bound rather than calling `zeros`
//!
//! Writing the case as a `Tensor::zeros` call works, but its diagnostic ends in
//! a `note: required by a bound in ...` that names the impl block, and rustc
//! renders that path differently depending on whether the `incin` facade is in
//! the dependency graph -- `impl Tensor<..>` under one feature set and
//! `impl incin::prelude::Tensor<..>` under another. The recorded output then
//! cannot match both CI configurations.
//!
//! Naming the bound directly pins the same property with a diagnostic that
//! points at this file. The positive half -- that `zeros` really does hand back
//! a `Dense` -- is covered by `typed_layout.rs`, which exercises it at runtime.

extern crate incin_core as incin;

use incin_core::shapes::{FreshDense, Layout, LayoutOf, Shape};
use incin_macros::s;

/// A downstream layout: legal to define, illegal to claim for an allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Sideways;

impl Layout for Sideways {}
impl<S: Shape> LayoutOf<S> for Sideways {}

/// Stands in for the bound every constructor carries.
fn only_a_layout_a_fresh_allocation_has<S: Shape, L: FreshDense<S>>() {}

fn main() {
    only_a_layout_a_fresh_allocation_has::<s![3, 4], Sideways>();
}
