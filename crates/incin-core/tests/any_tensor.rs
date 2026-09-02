//! `AnyTensor` should let generic code name one type parameter instead of six.
#![cfg(feature = "std")]

extern crate incin_core as incin;

use incin_backends::cpu::CpuBackendImpl;
use incin_core::prelude::*;
use incin_macros::s;

/// Without `AnyTensor`: six parameters and six bounds to call one method.
fn numel_spelled_out<S, B, K, G, P, L>(t: &Tensor<S, B, K, G, P, L>) -> usize
where
    S: incin_core::shapes::Shape + incin_core::shapes::DynShape,
    B: incin_core::backend_authoring::Backend,
    K: DType,
    G: incin_core::tensor::grad::RequiresGrad,
    P: Placement,
    L: Layout,
{
    t.numel()
}

/// With it: one parameter, and only the bound that is load-bearing.
fn numel_generic<T: AnyTensor>(t: &T) -> usize
where
    T::Shape: incin_core::shapes::DynShape,
{
    t.as_tensor().numel()
}

/// Both accept a proven tensor and an unproven one, which is the point: the
/// helper does not have to care, and does not have to name the layout to say so.
#[test]
fn any_tensor_collapses_the_parameter_list() {
    let plain = Tensor::<s![3, 4], CpuBackendImpl>::zeros(()).unwrap();
    let proven = Tensor::<s![3, 4], CpuBackendImpl>::zeros(())
        .unwrap()
        .into_row_major()
        .unwrap();

    assert_eq!(numel_spelled_out(&plain), 12);
    assert_eq!(numel_spelled_out(&proven), 12);
    assert_eq!(numel_generic(&plain), 12);
    assert_eq!(numel_generic(&proven), 12);
}

/// A bound that genuinely needs a parameter still reaches it, as an associated
/// type. Nothing is hidden -- only the parameters a helper does not constrain
/// stop having to be written down.
#[test]
fn a_needed_parameter_is_still_reachable() {
    fn is_dense<T: AnyTensor>(_: &T) -> bool
    where
        T::Layout: Contiguous,
    {
        true
    }

    let proven = Tensor::<s![3, 4], CpuBackendImpl>::zeros(())
        .unwrap()
        .into_row_major()
        .unwrap();
    assert!(is_dense(&proven));

    // is_dense(&plain) does not compile: `Unknown` is not `Contiguous`.
}
