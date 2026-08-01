//! The rustfmt fixture: one file invoking all three macros, kept in the exact
//! form `rustfmt` produces.
//!
//! `invoking_the_macros_leaves_a_file_formattable` formats this and asserts the
//! result is identical. If a macro's invocation form ever stops being
//! parseable as Rust, rustfmt skips it and this stops being a fixed point.
use ::incin::prelude::*;

#[module]
pub struct Formatted<B: Backend> {
    fc: Linear<s![8, 4], B>,
    head: Linear<s![4, 2], B>,
}

fn main() {
    let t = Tensor::<s![10, 20, 30]>::zeros(()).unwrap();
    let view = t.slice_idx::<idx![0..5, .., 15..30]>().unwrap();
    assert_eq!(view.dims().as_ref(), &[5, 20, 15]);

    let model = Formatted::<DefaultBackend> {
        fc: Linear::build(()).unwrap(),
        head: Linear::build(()).unwrap(),
    };
    assert!(!model.parameters().is_empty());

    type FormattedMesh = mesh![dp = 2, tp = 2, pp = 1];
    assert_eq!(<FormattedMesh as ::incin::dist::mesh::ValidMesh>::WORLD, 4);
}
