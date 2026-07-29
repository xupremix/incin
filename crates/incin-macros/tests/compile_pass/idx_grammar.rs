//! `idx!` accepts every form its parser declares.
use ::incin::prelude::*;

fn main() {
    let t = Tensor::<s![10, 20, 30]>::zeros(()).unwrap();

    let bounded = t.slice_idx::<idx![0..5, .., 15..30]>().unwrap();
    assert_eq!(bounded.dims().as_ref(), &[5, 20, 15]);

    let full = t.slice_idx::<idx![.., .., ..]>().unwrap();
    assert_eq!(full.dims().as_ref(), &[10, 20, 30]);

    // The reshape target forms: an inferred axis and a literal one.
    let flat = t.reshape_idx::<idx![-1]>().unwrap();
    assert_eq!(flat.dims().as_ref(), &[6000]);

    let split = t.reshape_idx::<idx![10, -1]>().unwrap();
    assert_eq!(split.dims().as_ref(), &[10, 600]);
}
