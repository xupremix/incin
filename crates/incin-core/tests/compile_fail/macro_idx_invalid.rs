extern crate incin_core as incin;
use incin_core as incin;
use incin_core::prelude::*;
use incin_core::prelude::dummy::DummyBackend;
use incin_macros::{s, idx};

fn main() {
    let t = Tensor::<s![10, 20], DummyBackend<f32, Cpu>>::zeros(()).unwrap();
    let _ = t.slice::<idx![..., -2]>();
}
