//! Integration coverage for `arbitrary_static_axis_literals_remain_static` on the documented public surface.
extern crate incin_core as incin;

use incin_core::advanced::ToAxisIndex;
use incin_core::prelude::axis;

#[test]
fn arbitrary_static_axis_literals_remain_static() {
    assert_eq!(ToAxisIndex::to_axis_index(&axis!(17)), 17);
    assert_eq!(ToAxisIndex::to_axis_index(&axis!(-17)), -17);
}
