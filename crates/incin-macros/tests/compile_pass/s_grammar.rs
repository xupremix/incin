//! `s!` accepts every form its parser declares, and each produces the shape it
//! says it does. A grammar with a branch nothing exercises is a branch that can
//! rot silently.
use ::incin::prelude::*;

::incin::dim!(Batch);

type Literals = s![2, 3, 4];
type Dynamic = s![dyn, 3];
type Underscore = s![_, 3];
type Named = s![Batch, 128];
type Repeat = s![4; 3];
type Tail = s![.., 3];
type Head = s![2, ..];
type Span = s![2, .., 4];

fn main() {
    let t = Tensor::<Literals>::zeros(()).unwrap();
    assert_eq!(t.dims().as_ref(), &[2, 3, 4]);

    let r = Tensor::<Repeat>::zeros(()).unwrap();
    assert_eq!(r.dims().as_ref(), &[4, 4, 4]);

    // The remaining forms are exercised for expansion rather than for a value:
    // a partially dynamic shape has no dims until one is supplied.
    let _: core::marker::PhantomData<(Dynamic, Underscore, Named, Tail, Head, Span)> =
        core::marker::PhantomData;
}
