use incin_macros::axes;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Batch;
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Channels;
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Height;
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct Width;

#[test]
fn axes_expansion_test() {
    type ImageAxes = axes![Batch, Channels, Height, Width];

    let instance: ImageAxes = (
        incin_core::shapes::NamedDyn::<Batch>::new(32),
        incin_core::shapes::NamedDyn::<Channels>::new(3),
        incin_core::shapes::NamedDyn::<Height>::new(224),
        incin_core::shapes::NamedDyn::<Width>::new(224),
    );

    assert_eq!(instance.0.size, 32);
    assert_eq!(instance.1.size, 3);
    assert_eq!(instance.2.size, 224);
    assert_eq!(instance.3.size, 224);
}

#[test]
fn axes_compile_fail_diagnostics() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/axes_compile_fail/*.rs");
}
