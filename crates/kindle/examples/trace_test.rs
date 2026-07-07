use kindle::prelude::*;

use typenum::{Prod, U2, U3};
use std::any::TypeId;

fn main() -> Result<()> {
    let type_a = TypeId::of::<Prod<U2, U3>>();
    let type_b = TypeId::of::<Prod<U3, U2>>();
    
    assert_eq!(type_a, type_b);
    println!("They are exactly the same type!");
    
    Ok(())
}
