extern crate incin_core as incin;

use incin_core::prelude::{BroadcastShape, Shape};
use incin_macros::s;

incin_core::dim!(Batch, Channels);

fn require_broadcast<L, R>()
where
    L: Shape + BroadcastShape<R>,
    R: Shape,
{
}

fn main() {
    require_broadcast::<s![Batch], s![Channels]>();
}
