#[test]
/// Core abstraction for `compile_fail` within the Kindle framework.
fn compile_fail() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/*.rs");
}
