//! Integration coverage for `main` on the documented public surface.
extern crate incin_core as incin;
use incin_core::prelude::*;
use incin_macros::s;

fn requires_matmul<L, R>()
where
    L: MatMulShape<R>,
    R: Shape,
{
}

fn main() {
    // The rank-8 rule exists, but 2 cannot contract with 4.
    requires_matmul::<
        s![1, 1, 1, 1, 1, 1, 3, 2],
        s![1, 1, 1, 1, 1, 1, 4, 5],
    >();
}
