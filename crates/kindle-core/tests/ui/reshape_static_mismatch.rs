use kindle_core::prelude::*;
use kindle_core::tensor::backend::dummy::DummyBackend;
use kindle_macros::s;
use typenum::{U2, U3, U4, U6};

fn main() {
    let t = Tensor::<s![U2, U3], DummyBackend>::zeros(()).unwrap();
    
    // Reshaping to (U6,) is correct, but let's reshape to (U2, U4) which is 8 elements instead of 6.
    let _ = t.reshape::<s![U2, U4]>(((), ()));
}
