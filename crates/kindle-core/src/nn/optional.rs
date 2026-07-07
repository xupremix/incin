use crate::prelude::Dyn;

/// Marker trait for optional module parameters (e.g., bias tensors).
///
/// Three types implement this trait:
/// * [`True`]  — The parameter always exists. `new_with` takes no extra arg.
/// * [`False`] — The parameter never exists. `new_with` takes no extra arg.
/// * [`Dyn`]   — Whether the parameter exists is decided at runtime via a `bool` passed to `new_with`.
///
/// There is no `build()` method or associated `BuildArgs` — each layer provides
/// dedicated `impl` blocks with the exact signature matching each variant.
pub trait OptionalField {}

/// The parameter should **always** exist.
#[derive(Debug, Clone, Copy, Default)]
pub struct True;
impl OptionalField for True {}

/// The parameter should **never** exist.
#[derive(Debug, Clone, Copy, Default)]
pub struct False;
impl OptionalField for False {}

/// The parameter's existence is decided at runtime (pass a `bool` to `new_with`).
/// This reuses the existing [`Dyn`] struct from the prelude.
impl OptionalField for Dyn {}
