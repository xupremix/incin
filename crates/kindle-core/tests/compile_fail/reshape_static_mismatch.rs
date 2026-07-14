extern crate kindle_core as kindle;
use kindle_core as kindle;
use kindle_core::prelude::*;
use kindle_core::tensor::backend::dummy::DummyBackend;
use kindle_macros::s;
use typenum::{typenum::U2, typenum::U3, typenum::U4, typenum::U6};

fn main() {
    let t = Tensor::<s![2, 3], DummyBackend<f32, Cpu>>::zeros(()).unwrap();
    
    let _ = t.reshape::<s![2, 4]>(((), ()));
}
