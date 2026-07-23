extern crate kindle_core as kindle;
use kindle_core::prelude::*;
use kindle_core::prelude::dummy::DummyBackend;
use kindle_macros::s;

/// Backend.
type Backend = DummyBackend<f32, kindle_core::prelude::Cpu>;

fn main() {
    let t = Tensor::<s![2, 4, 32, 32], Backend>::zeros(()).unwrap();
    let w = Tensor::<s![8, 3, 3, 3], Backend>::zeros(()).unwrap();

    // Input channels (4) != kernel input channels (3); spatial size (32) is
    // large enough that this is purely a channel mismatch, distinct from the
    // arithmetic-underflow case in conv2d_invalid_shape.rs.
    let _c = t.conv2d::<typenum::U1, typenum::U1, s![8, 3, 3, 3]>(&w, None).unwrap();
}
