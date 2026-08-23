//! Example: exercises the documented API around `main`.
use incin::prelude::*;

use std::any::TypeId;
use typenum::{Prod, U2, U3};

fn main() -> Result<()> {
    let type_a = TypeId::of::<Prod<U2, U3>>();
    let type_b = TypeId::of::<Prod<U3, U2>>();

    assert_eq!(type_a, type_b);
    println!("They are exactly the same type!");

    Ok(())
}
