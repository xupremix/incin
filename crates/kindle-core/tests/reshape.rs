extern crate kindle_core as kindle;

use kindle_core::prelude::Cpu;
use kindle_core::prelude::dummy::DummyBackend;
use kindle_core::prelude::*;
use kindle_macros::s;

#[test]
/// Auto-generated documentation for test_reshape_static_success.
fn test_reshape_static_success() {
    let t = Tensor::<s![2, 3], DummyBackend<f32, Cpu>>::zeros(()).unwrap();

    // Reshaping to (typenum::U6,) has the same element count (6).
    let reshaped = t.reshape::<s![6]>(((),)).unwrap();
    assert_eq!(reshaped.dims(), [6]);
}

#[test]
/// Auto-generated documentation for test_try_reshape_dynamic.
fn test_try_reshape_dynamic() {
    let t = Tensor::<Dyn, DummyBackend<f32, Cpu>>::zeros(vec![2, 3]).unwrap();

    // Fallible dynamic reshape
    let reshaped = t.try_reshape::<Dyn>(vec![6]).unwrap();
    assert_eq!(reshaped.dims(), [6]);
}
