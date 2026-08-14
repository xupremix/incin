//! The expansions must not depend on what the caller imported, or on the name
//! they imported it under.
//!
//! No `use ::incin::prelude::*` anywhere below. Each macro is named through an
//! alias, and everything the expansion needs it must reach on its own.
use ::incin as renamed;
use renamed::prelude::Backend;
use renamed::prelude::{idx, module, s, VariableBackend};

type Shape = s![2, 3];

#[module]
pub struct Renamed<B: Backend + VariableBackend> {
    fc: renamed::prelude::Linear<s![8, 4], B>,
}

fn main() {
    let t = renamed::prelude::Tensor::<Shape>::zeros(()).unwrap();
    assert_eq!(t.dims().as_ref(), &[2, 3]);

    let big = renamed::prelude::Tensor::<s![10, 20, 30]>::zeros(()).unwrap();
    let view = big.slice_idx::<idx![0..5, .., 15..30]>().unwrap();
    assert_eq!(view.dims().as_ref(), &[5, 20, 15]);

    let m = Renamed::<renamed::prelude::DefaultBackend> {
        fc: renamed::prelude::Linear::build(()).unwrap(),
    };
    // `parameters()` comes from a trait the expansion implements; the caller
    // has to have it in scope, which is the one thing a macro cannot do for
    // them without also polluting it.
    use renamed::prelude::Parameters;
    assert!(!m.parameters().is_empty());
}
