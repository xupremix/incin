use incin_core::dist::mesh::{ValidMesh, mesh};

#[test]
fn test_mesh_macro_expansion() {
    type TestMesh1 = mesh![dp = 2, tp = 4, pp = 1];
    assert_eq!(TestMesh1::DATA, 2);
    assert_eq!(TestMesh1::TENSOR, 4);
    assert_eq!(TestMesh1::PIPELINE, 1);
    assert_eq!(TestMesh1::WORLD, 8);

    // Default parameters (omitted tp and pp default to 1)
    type TestMesh2 = mesh![dp = 3];
    assert_eq!(TestMesh2::DATA, 3);
    assert_eq!(TestMesh2::TENSOR, 1);
    assert_eq!(TestMesh2::PIPELINE, 1);
    assert_eq!(TestMesh2::WORLD, 3);

    // Alternative keyword names
    type TestMesh3 = mesh![data = 2, tensor = 2, pipeline = 2];
    assert_eq!(TestMesh3::DATA, 2);
    assert_eq!(TestMesh3::TENSOR, 2);
    assert_eq!(TestMesh3::PIPELINE, 2);
    assert_eq!(TestMesh3::WORLD, 8);
}
