//! Integration coverage for `main` on the documented public surface.
use incin_core::dist::mesh::mesh;

fn main() {
    type InvalidMesh = mesh![dp = 0];
}
