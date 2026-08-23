//! Integration coverage for `test_reshape_static_success` on the documented public surface.
extern crate incin_core as incin;

use incin_backends::cpu::CpuBackendImpl;
use incin_core::prelude::*;
use incin_macros::s;

#[test]
/// Test reshape static success.
fn test_reshape_static_success() {
    let t = Tensor::<s![2, 3], CpuBackendImpl>::zeros(()).unwrap();

    // Reshaping to (typenum::U6,) has the same element count (6).
    let reshaped = t.reshape(shape![6]).unwrap();
    let dims: &[usize] = reshaped.shape_buf().as_ref();
    assert_eq!(dims, &[6]);
}

#[test]
/// Test try reshape dynamic.
fn test_try_reshape_dynamic() {
    let t = Tensor::<Dyn, CpuBackendImpl>::zeros(vec![2, 3]).unwrap();

    // Fallible dynamic reshape
    let reshaped = t.try_reshape::<Dyn>(vec![6]).unwrap();
    let dims = reshaped.dims();
    assert_eq!(dims.as_ref(), &[6]);
}
