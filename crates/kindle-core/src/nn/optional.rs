use crate::prelude::Dyn;

/// Marker trait for optional module parameters (e.g., bias tensors).
///
/// Three types implement this trait:
/// * [`True`]  — The parameter always exists. `build` takes no extra arg.
/// * [`False`] — The parameter never exists. `build` takes no extra arg.
/// * [`Dyn`]   — Whether the parameter exists is decided at runtime via a `bool` passed to `build`.
///
/// There is no `build()` method or associated `BuildArgs` — each layer provides
/// dedicated `impl` blocks with the exact signature matching each variant.
pub trait OptionalField {
    /// Runtime argument contributed by this optional position.
    type Arg;
    /// Resolves whether the field is present.
    fn init(arg: Self::Arg) -> bool;
}

/// The parameter should **always** exist.
#[derive(Debug, Clone, Copy, Default)]
pub struct True;
impl OptionalField for True {
    type Arg = ();
    fn init(_: ()) -> bool {
        true
    }
}

/// The parameter should **never** exist.
#[derive(Debug, Clone, Copy, Default)]
pub struct False;
impl OptionalField for False {
    type Arg = ();
    fn init(_: ()) -> bool {
        false
    }
}

/// The parameter's existence is decided at runtime (pass a `bool` to `build`).
/// This reuses the existing [`Dyn`] struct from the prelude.
impl OptionalField for Dyn {
    type Arg = bool;
    fn init(arg: bool) -> bool {
        arg
    }
}
