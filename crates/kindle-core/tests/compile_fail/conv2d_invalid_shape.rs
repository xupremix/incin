extern crate kindle_core as kindle;
use kindle_core as kindle;
use kindle_core::prelude::*;
use kindle_macros::s;
use kindle_core::prelude::dummy::DummyBackend;

type Backend = DummyBackend<f32, kindle_core::prelude::Cpu>;

fn main() {
    let t = Tensor::<s![1, 3, 2, 2], Backend>::zeros(()).unwrap();
    let w = Tensor::<s![8, 3, 3, 3], Backend>::zeros(()).unwrap();
    
    // Kernel larger than image with no padding causes negative shape -> compile error!
    let _c = t.conv2d::<typenum::U1, typenum::U0, s![8, 3, 3, 3]>(&w, None).unwrap();
}
