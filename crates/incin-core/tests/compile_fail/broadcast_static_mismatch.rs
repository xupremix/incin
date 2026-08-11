extern crate incin_core as incin;

use incin::prelude::*;

type B = incin::test_utils::DummyBackend<incin::prelude::Cpu>;

fn main() {
    let lhs = Tensor::<s![32], B>::ones(()).unwrap();
    let rhs = Tensor::<s![64], B>::ones(()).unwrap();
    // Static broadcast incompatibility must be rejected at the canonical
    // operation boundary, rather than deferred to a runtime shape error.
    let _ = lhs.broadcast_add(&rhs);
}
