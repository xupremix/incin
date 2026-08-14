use incin_core::prelude::{Dim, ShapeError};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Forged;

impl Dim for Forged {
    type KeepDim = incin_core::typenum::U1;
    type Arg = ();

    fn resolve_arg(_: Self::Arg) -> Result<usize, ShapeError> { Ok(1) }
}

fn main() {}
