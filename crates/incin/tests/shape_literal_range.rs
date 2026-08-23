//! Integration coverage for `shape_literal_range_uses_raw_static_extent_types` on the documented public surface.
use incin::prelude::*;

#[test]
fn shape_literal_range_uses_raw_static_extent_types() {
    let value = shape![0, 1, 64, 4096, 65_536, 1_000_000];
    assert_eq!(
        value.shape_buf().as_ref(),
        &[0, 1, 64, 4096, 65_536, 1_000_000]
    );
}
