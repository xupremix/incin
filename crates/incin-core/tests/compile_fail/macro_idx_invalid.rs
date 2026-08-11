extern crate incin_core as incin;
use incin_core::prelude::*;
use incin_core::test_utils::DummyBackend;
use incin_macros::{s, idx};

fn main() {
    let t = Tensor::<s![10, 20], DummyBackend<Cpu>>::zeros(()).unwrap();
    let _ = t.slice::<idx![..., -2]>();
}
