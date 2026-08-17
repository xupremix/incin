//! `SHP-007`: constructor arguments resolve through the same rank ceiling as
//! `Shape`; a typed shape must not become unconstructible at rank 6 or 8.

extern crate incin_core as incin;

use incin_backends::cpu::CpuBackendImpl;
use incin_core::prelude::{Dyn, Tensor, s};

type Backend = CpuBackendImpl;

#[test]
fn fully_static_shapes_construct_at_every_previously_missing_rank() {
    #[allow(clippy::type_complexity)]
    let rank6: Tensor<s![1, 1, 1, 1, 1, 1], Backend> = Tensor::zeros(()).unwrap();
    #[allow(clippy::type_complexity)]
    let rank8: Tensor<s![1, 1, 1, 1, 1, 1, 1, 1], Backend> = Tensor::zeros(()).unwrap();

    assert_eq!(rank6.dims().as_ref(), &[1, 1, 1, 1, 1, 1]);
    assert_eq!(rank8.dims().as_ref(), &[1, 1, 1, 1, 1, 1, 1, 1]);
}

#[test]
fn a_dynamic_shape_reports_the_rank_element_count_and_dims_it_was_built_with() {
    let t: Tensor<Dyn, Backend> = Tensor::zeros(vec![2, 3]).unwrap();
    assert_eq!(t.rank(), 2);
    assert_eq!(t.numel(), 6);
    assert_eq!(t.dims(), vec![2, 3]);
}

#[test]
fn ones_agrees_with_zeros_on_shape_and_fills_with_ones() {
    let t: Tensor<Dyn, Backend> = Tensor::ones(vec![4]).unwrap();
    assert_eq!(t.rank(), 1);
    assert_eq!(t.numel(), 4);
    // A shape-only backend could not have told these two constructors apart;
    // the real backend can, so the fill value is asserted too.
    assert_eq!(t.to_vec1::<f32>().unwrap(), vec![1.0f32; 4]);
}
