use kindle_core::prelude::*;
use kindle_core::tensor::backend::dummy::DummyBackend;
use typenum::{U1, U2, U3};

fn main() {
    let t1: Tensor<(U1, U2), DummyBackend> = Tensor::zeros(()).unwrap();
    let t2: Tensor<(U3, U3), DummyBackend> = Tensor::zeros(()).unwrap();
    
    // Broadcast of (1, 2) and (3, 3) should fail since 2 != 3 and neither is 1!
    let _out = t1.broadcast_add(&t2).unwrap();
}
