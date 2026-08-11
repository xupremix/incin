extern crate incin_core as incin;

use incin_core::test_utils::DummyBackend;
use incin_core::prelude::*;

fn main() {
    type B = DummyBackend<Cpu>;
    let _ = Linear::<Dyn, B, Dyn>::build((true, 10, 20));
}
