extern crate kindle_core as kindle;
use kindle_core as kindle;
use kindle_core::prelude::*;
use kindle_core::prelude::dummy::DummyBackend;
use kindle_macros::{s, idx};

fn main() {
    let t = Tensor::<s![10, 20], DummyBackend<f32, Cpu>>::zeros(()).unwrap();
    let _ = t.slice::<idx![..., -2]>();
}
