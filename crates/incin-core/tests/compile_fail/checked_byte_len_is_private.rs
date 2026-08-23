//! Integration coverage for `main` on the documented public surface.
use incin_core::prelude::CheckedByteLen;

fn main() {
    let _forged = CheckedByteLen(16);
}
