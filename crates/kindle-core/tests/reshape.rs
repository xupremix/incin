extern crate kindle_core as kindle;

use kindle_core::prelude::*;
use kindle_core::tensor::backend::dummy::DummyBackend;
use kindle_core::tensor::device::Cpu;
use kindle_macros::s;


#[test]
fn test_reshape_static_success() {
    let t = Tensor::<s![2, 3], DummyBackend<f32, Cpu>>::zeros(()).unwrap();

    // Reshaping to (typenum::U6,) has the same element count (6).
    let reshaped = t.reshape::<s![6]>(((),)).unwrap();
    assert_eq!(reshaped.dims(), [6]);
}

#[test]
fn test_try_reshape_dynamic() {
    let t = Tensor::<Dyn, DummyBackend<f32, Cpu>>::zeros(vec![2, 3]).unwrap();

    // Fallible dynamic reshape
    let reshaped = t.try_reshape::<Dyn>(vec![6]).unwrap();
    assert_eq!(reshaped.dims(), [6]);
}
