extern crate incin_core as incin;

use incin_core::prelude::dummy::DummyBackend;
use incin_core::prelude::*;

fn main() {
    type B = DummyBackend<f32, Cpu>;
    let _ = Linear::<Dyn, B>::build((10, 20, 30));
}
