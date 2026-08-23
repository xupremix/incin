//! Integration coverage for `Batch` on the documented public surface.
use incin::prelude::*;
use incin::prelude::axis;

incin::dim!(Batch, Channels);

fn main() {
    type S = s![Batch, Channels];
    let selector = axis!(named Channels);
    assert_eq!(selector.resolve::<S>().unwrap(), 1);
}
