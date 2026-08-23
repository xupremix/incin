//! Integration coverage for `main` on the documented public surface.
use incin_macros::mesh;

fn main() {
    type InvalidKey = mesh![invalid = 4];
}
