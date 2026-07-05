extern crate kindle_core as kindle;
use kindle_core::prelude::*;
use kindle_core::prelude::dummy::DummyBackend;

type Backend = DummyBackend<f32, kindle_core::tensor::device::Cpu>;

fn main() {
    let t = Tensor::<(Const<1>, Const<3>, Const<16>, Const<16>), Backend>::zeros(()).unwrap();
    let w = Tensor::<(Const<8>, Const<3>, Const<3>, Const<3>), Backend>::zeros(()).unwrap();
    
    // Invalid padding size for valid convolution!
    let _c = t.conv2d::<typenum::U1, typenum::U3, (Const<8>, Const<3>, Const<3>, Const<3>)>(&w, None).unwrap();
}
