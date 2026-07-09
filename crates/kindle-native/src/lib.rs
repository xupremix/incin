//! # Kindle Native
//!
//! `kindle-native` is a pure-Rust, ownership-flavored, CPU-only implementation
//! of the `Backend` trait defined in `kindle-core`. It provides its own
//! strided-view tensor storage (`storage`) and shape/stride math (`stride`),
//! independent of any external tensor compute library (Candle, ndarray, burn).
//!
//! This crate is built incrementally across multiple phases; this initial
//! skeleton wires up the crate and its foundational shape/storage modules.
//! The `NativeBackend` type and its `Backend` trait implementation land in a
//! later phase once the autograd tape and variable types exist.

pub use kindle_core::prelude::*;

pub mod storage;
pub mod stride;
pub mod tape;
pub mod var;
