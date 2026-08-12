extern crate incin_core as incin;

use incin::prelude::*;

type B = incin::test_utils::DummyBackend<incin::prelude::Cpu>;

fn main() {
    // Static broadcast incompatibility must be rejected at the canonical
    // operation boundary, rather than deferred to a runtime shape error.
    assert_broadcast::<s![32], s![64]>();
}

fn assert_broadcast<L, R>()
where
    L: Shape + BroadcastShape<R>,
    R: Shape,
    <L as BroadcastShape<R>>::Output: Shape + DynShape,
{
}
