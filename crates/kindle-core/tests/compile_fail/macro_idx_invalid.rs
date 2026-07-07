use kindle_core::prelude::*;
use kindle_core::prelude::dummy::DummyBackend;
use kindle_macros::{s, idx};

fn main() {
    let t = Tensor::<s![10, 20], DummyBackend>::zeros(()).unwrap();
    let _ = t.slice::<idx![..., -2]>();
}
