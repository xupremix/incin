extern crate incin_core as incin;

use incin_core::prelude::{axis, ToAxisIndex};

#[test]
fn arbitrary_static_axis_literals_remain_static() {
    assert_eq!(ToAxisIndex::to_axis_index(&axis!(17)), 17);
    assert_eq!(ToAxisIndex::to_axis_index(&axis!(-17)), -17);
}
