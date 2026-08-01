use incin_core::dist::mesh::{ValidMesh, mesh};

fn main() {
    type M1 = mesh![dp = 2, tp = 4, pp = 1];
    type M2 = mesh![dp = 2];
    type M3 = mesh![data = 1, tensor = 2, pipeline = 4];
    assert_eq!(M1::WORLD, 8);
    assert_eq!(M2::WORLD, 2);
    assert_eq!(M3::WORLD, 8);
}
