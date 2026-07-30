use incin_core::dist::mesh::ValidMesh;
use incin_macros::mesh;

#[test]
fn mesh_expansion_defaults_and_explicit_axes() {
    // Single axis specified, defaults dp=3, tp=1, pp=1
    type DpOnly = mesh![dp = 3];
    assert_eq!(DpOnly::DATA, 3);
    assert_eq!(DpOnly::TENSOR, 1);
    assert_eq!(DpOnly::PIPELINE, 1);
    assert_eq!(DpOnly::WORLD, 3);

    // Two axes specified, dp=2, tp=4, pp=1
    type DpTp = mesh![dp = 2, tp = 4];
    assert_eq!(DpTp::DATA, 2);
    assert_eq!(DpTp::TENSOR, 4);
    assert_eq!(DpTp::PIPELINE, 1);
    assert_eq!(DpTp::WORLD, 8);

    // All three axes specified, dp=2, tp=2, pp=2
    type AllAxes = mesh![dp = 2, tp = 2, pp = 2];
    assert_eq!(AllAxes::DATA, 2);
    assert_eq!(AllAxes::TENSOR, 2);
    assert_eq!(AllAxes::PIPELINE, 2);
    assert_eq!(AllAxes::WORLD, 8);
}

#[test]
fn mesh_compile_fail_diagnostics() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/mesh_compile_fail/*.rs");
}
