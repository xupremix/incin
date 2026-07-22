extern crate kindle_core as kindle;

use kindle_core::prelude::dummy::DummyBackend;
use kindle_core::prelude::*;

fn main() {
    type B = DummyBackend<f32, Cpu>;
    let _ = Linear::<Dyn, B>::build((10, 20, 30));
}
