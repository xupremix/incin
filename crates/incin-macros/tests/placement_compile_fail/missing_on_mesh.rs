//! Integration coverage for `main` on the documented public surface.
use incin_macros::placement;

fn main() {
    type Invalid = placement![Replicated];
}
