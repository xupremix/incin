//! `SHP-007`: constructor arguments resolve through the same rank ceiling as
//! `Shape`; a typed shape must not become unconstructible at rank 6 or 8.

use incin_core::prelude::dummy::DummyBackend;
use incin_core::prelude::{Cpu, Tensor};
use typenum::U1;

type Backend = DummyBackend<f32, Cpu>;

#[test]
fn fully_static_shapes_construct_at_every_previously_missing_rank() {
    let rank6: Tensor<(U1, U1, U1, U1, U1, U1), Backend> = Tensor::zeros(()).unwrap();
    let rank8: Tensor<(U1, U1, U1, U1, U1, U1, U1, U1), Backend> = Tensor::zeros(()).unwrap();

    assert_eq!(rank6.dims().as_ref(), &[1, 1, 1, 1, 1, 1]);
    assert_eq!(rank8.dims().as_ref(), &[1, 1, 1, 1, 1, 1, 1, 1]);
}
