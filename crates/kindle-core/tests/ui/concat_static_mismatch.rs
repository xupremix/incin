use kindle_core::prelude::*;
use kindle_core::tensor::backend::dummy::DummyBackend;

fn main() {
    let t1: Tensor<s![U2, U3], DummyBackend> = Tensor::zeros([2, 3]).unwrap();
    let t2: Tensor<s![U4, U3], DummyBackend> = Tensor::zeros([4, 3]).unwrap();

    // Trying to concatenate along axis 1 (the U3 axis) fails because U2 != U4
    let _ = t1.concat::<s![U4, U3], U1>(&t2);
}
