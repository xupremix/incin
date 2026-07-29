//! `idx!` accepts `start..end` and a bare `..`, and nothing between: a
//! half-open range has no static extent to put in the type.
use ::incin::prelude::*;

fn main() {
    let t = Tensor::<s![10, 20]>::zeros(()).unwrap();
    let _ = t.slice_idx::<idx![0.., ..]>();
}
