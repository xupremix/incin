extern crate incin_core as incin;

use incin_core::prelude::Cpu;
use incin_core::prelude::dummy::DummyBackend;
use incin_core::prelude::*;
use incin_macros::s;

#[test]
/// Test reshape static success.
fn test_reshape_static_success() {
    let t = Tensor::<s![2, 3], DummyBackend<f32, Cpu>>::zeros(()).unwrap();

    // Reshaping to (typenum::U6,) has the same element count (6).
    let reshaped = t.reshape::<s![6]>(((),)).unwrap();
    assert_eq!(reshaped.dims(), [6]);
}

#[test]
/// Test try reshape dynamic.
fn test_try_reshape_dynamic() {
    let t = Tensor::<Dyn, DummyBackend<f32, Cpu>>::zeros(vec![2, 3]).unwrap();

    // Fallible dynamic reshape
    let reshaped = t.try_reshape::<Dyn>(vec![6]).unwrap();
    assert_eq!(reshaped.dims(), [6]);
}
