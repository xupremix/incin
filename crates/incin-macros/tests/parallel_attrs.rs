use incin::prelude::*;
use incin::VariableBackend;
use incin_macros::mesh;

#[allow(dead_code)]
type MyMesh = mesh![dp = 2, tp = 4];

#[module]
pub struct ParallelModel<B: Backend + VariableBackend> {
    #[parallel(mesh = MyMesh, stage = 0)]
    layer1: Linear<s![768, 256], B>,

    #[shard(mesh = MyMesh, axis = 0)]
    layer2: Linear<s![256, 10], B>,
}

#[test]
fn parallel_and_shard_field_attributes_pass() {
    let model = ParallelModel::<DefaultBackend> {
        layer1: Linear::build(()).unwrap(),
        layer2: Linear::build(()).unwrap(),
    };

    assert_eq!(model.parameters().len(), 4);
}

#[test]
fn parallel_attrs_compile_fail_diagnostics() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/parallel_attrs_compile_fail/*.rs");
}
