use incin_macros::parallel;

#[test]
fn test_parallel_block_simple() {
    let val = parallel!({
        10 + 20
    });
    assert_eq!(val, 30);
}

#[test]
fn test_parallel_block_with_mesh() {
    let dummy_mesh = "mesh_spec";
    let val = parallel!(dummy_mesh => {
        42
    });
    assert_eq!(val, 42);
}
