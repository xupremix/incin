//! `idx!` takes a simple identifier for a named dimension. A qualified path is
//! rejected by the macro rather than expanded into a path that will not
//! resolve.
use ::incin::prelude::*;

fn main() {
    let t = Tensor::<s![10, 20]>::zeros(()).unwrap();
    let _ = t.slice_idx::<idx![some::Module::Batch, ..]>();
}
