extern crate incin_core as incin;
use incin_core::prelude::*;
use incin_core::test_utils::DummyBackend;
use incin_macros::s;

fn main() {
    let t = Tensor::<s![2, 3], DummyBackend<f32, Cpu>>::zeros(()).unwrap();
    
    let _ = t.reshape::<s![2, 4]>(((), ()));
}
