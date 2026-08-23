//! Integration coverage for `leaked` on the documented public surface.
// Without the `test-utils` feature the module itself must not exist.
use incin::test_utils::fail_assign_on;

pub fn leaked() {
    let _ = fail_assign_on;
}
