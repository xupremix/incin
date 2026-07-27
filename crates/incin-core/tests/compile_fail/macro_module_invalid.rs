//! `#[module]` recognizes exactly one field argument, `ignore`. Anything else
//! is a typo for it, and a typo that compiled would silently change which
//! fields are treated as parameters.
//!
//! This case previously wrote `#[module(foo = "bar")]` on the *struct*, which
//! the macro accepts and discards, so the file's only error came from an
//! unrelated duplicate import and it asserted nothing about the macro at all.
//! The struct-level argument is still accepted silently; that gap belongs to
//! `CI-005`, which owns the macro hygiene suite.

// The macro aborts before the struct is expanded, so its imports look unused.
#![allow(unused_imports)]

extern crate incin_core as incin;
use incin_core::prelude::*;
use incin_macros::{module, s};

#[module]
/// My net.
pub struct MyNet<B: Backend> {
    /// Linear.
    #[module(ignor)]
    pub linear: Linear<s![10, 10], B>,
}

fn main() {}
