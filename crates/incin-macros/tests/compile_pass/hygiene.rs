//! The expansions must not resolve against names the *caller* has in scope.
//!
//! Every decoy here is a name one of the three macros emits. Before `CI-005`
//! the `incin` module below broke all three: they expanded to a relative
//! `incin::prelude::…`, so a caller who happened to have that name won, and
//! the failure surfaced as `cannot find typenum in prelude` pointing at the
//! caller's own macro invocation.
use ::incin::prelude::*;

/// The decoy that used to win.
mod incin {
    pub mod prelude {
        pub struct Decoy;
    }
}

/// `s!` emits `typenum::UInt`, `typenum::UTerm`, `typenum::B0`, `typenum::B1`.
mod typenum {
    pub struct UTerm;
    pub struct B0;
    pub struct B1;
    pub struct UInt<A, B>(A, B);
}

/// `idx!` emits these three by name.
struct Ellipsis;
struct InferDim;
struct Slice<A, B, C>(A, B, C);

/// `#[module]` emits `Vec` and `format!` through the prelude path.
#[allow(dead_code)]
struct Vec;

#[module]
pub struct Shadowed<B: ::incin::prelude::Backend> {
    fc: Linear<s![8, 4], B>,
}

fn main() {
    let t = ::incin::prelude::Tensor::<s![2, 3]>::zeros(()).unwrap();
    assert_eq!(t.dims().as_ref(), &[2, 3]);

    let big = ::incin::prelude::Tensor::<s![10, 20, 30]>::zeros(()).unwrap();
    let view = big.slice_idx::<idx![0..5, .., 15..30]>().unwrap();
    assert_eq!(view.dims().as_ref(), &[5, 20, 15]);

    let m = Shadowed::<::incin::prelude::DefaultBackend> {
        fc: ::incin::prelude::Linear::build(()).unwrap(),
    };
    assert!(!m.parameters().is_empty());

    // The decoys are still the caller's, which is the other half of hygiene:
    // the macro must not capture them either.
    let _: incin::prelude::Decoy = incin::prelude::Decoy;
    let _: typenum::UTerm = typenum::UTerm;
    let _: Ellipsis = Ellipsis;
    let _: InferDim = InferDim;
    let _: Slice<u8, u8, u8> = Slice(0, 0, 0);
}
