use kindle_core::prelude::*;
use kindle_core::tensor::backend::dummy::DummyBackend;
use kindle_macros::s;
use typenum::{U2, U3, U6};

#[test]
fn test_reshape_static_success() {
    let t = Tensor::<s![U2, U3], DummyBackend>::zeros(()).unwrap();
    
    // Reshaping to (U6,) has the same element count (6).
    let reshaped = t.reshape::<s![U6]>(((),)).unwrap();
    assert_eq!(reshaped.dims(), [6]);
}

#[test]
fn test_try_reshape_dynamic() {
    let t = Tensor::<Dyn, DummyBackend>::zeros(vec![2, 3]).unwrap();
    
    // Fallible dynamic reshape
    let reshaped = t.try_reshape::<Dyn>(vec![6]).unwrap();
    assert_eq!(reshaped.dims(), [6]);
}
