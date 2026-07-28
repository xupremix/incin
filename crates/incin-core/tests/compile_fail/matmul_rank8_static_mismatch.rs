use incin_core::prelude::*;
use typenum::{U1, U2, U3, U4, U5};

fn requires_matmul<L, R>()
where
    L: MatMulShape<R>,
    R: Shape,
{
}

fn main() {
    // The rank-8 rule exists, but 2 cannot contract with 4.
    requires_matmul::<
        (U1, U1, U1, U1, U1, U1, U3, U2),
        (U1, U1, U1, U1, U1, U1, U4, U5),
    >();
}
