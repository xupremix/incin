//! Integration coverage for `Forged` on the documented public surface.
use incin_core::prelude::{ConcreteStaticExtent, Dim};

fn require<T: ConcreteStaticExtent>() {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Forged;

fn main() {
    require::<Forged>();
}

impl Dim for Forged {
    type KeepDim = incin_core::typenum::U1;
    type Arg = ();
}

impl ConcreteStaticExtent for Forged {
    type Nat = incin_core::typenum::U1;
}
