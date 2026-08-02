//! `EXE-009`: an operation family a backend does not implement must not be
//! answerable by silently inheriting an unsupported default.
//!
//! `ModuleOps` used to give every method a body returning
//! `UnsupportedBackendOperation`, so this exact empty impl compiled. That is
//! how `DispatchBackend` came to advertise normalization through the capability
//! registry while refusing `layer_norm` at run time: nothing in the type system
//! objected. With the defaults gone the omission is an `E0046`, which is where
//! a missing operation belongs.

use incin_core::backend_authoring::{Backend, ModuleOps};

struct Incomplete<B>(core::marker::PhantomData<B>);

impl<B: Backend> ModuleOps<B> for Incomplete<B> {}

fn main() {}
