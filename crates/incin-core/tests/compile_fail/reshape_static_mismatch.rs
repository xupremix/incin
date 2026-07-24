extern crate incin_core as incin;
use incin_core as incin;
use incin_core::prelude::*;
use incin_core::prelude::dummy::DummyBackend;
use incin_macros::s;
use typenum::{typenum::U2, typenum::U3, typenum::U4, typenum::U6};

fn main() {
    let t = Tensor::<s![2, 3], DummyBackend<f32, Cpu>>::zeros(()).unwrap();
    
    let _ = t.reshape::<s![2, 4]>(((), ()));
}
