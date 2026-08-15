//! `SHP-007`: constructor arguments resolve through the same rank ceiling as
//! `Shape`; a typed shape must not become unconstructible at rank 6 or 8.

extern crate incin_core as incin;

use incin_core::prelude::{Cpu, Tensor, s};
use incin_core::test_utils::DummyBackend;

type Backend = DummyBackend<Cpu>;

#[test]
fn fully_static_shapes_construct_at_every_previously_missing_rank() {
    #[allow(clippy::type_complexity)]
    let rank6: Tensor<s![1, 1, 1, 1, 1, 1], Backend> = Tensor::zeros(()).unwrap();
    #[allow(clippy::type_complexity)]
    let rank8: Tensor<s![1, 1, 1, 1, 1, 1, 1, 1], Backend> = Tensor::zeros(()).unwrap();

    assert_eq!(rank6.dims().as_ref(), &[1, 1, 1, 1, 1, 1]);
    assert_eq!(rank8.dims().as_ref(), &[1, 1, 1, 1, 1, 1, 1, 1]);
}
