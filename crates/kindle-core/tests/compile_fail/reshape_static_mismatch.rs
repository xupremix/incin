use kindle_core::prelude::*;
use kindle_core::tensor::backend::dummy::DummyBackend;
use kindle_macros::s;
use typenum::{U2, U3, U4, U6};

fn main() {
    let t = Tensor::<s![U2, U3], DummyBackend<f32, Cpu>>::zeros(()).unwrap();
    
    let _ = t.reshape::<s![U2, U4]>(((), ()));
}
