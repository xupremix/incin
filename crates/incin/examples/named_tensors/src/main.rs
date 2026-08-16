//! Minimal smoke test for `dim!` — for a fuller tour (transpose and
//! reshape safety, dimension unwrapping), run `cargo run -p named_dims_safety`.
extern crate alloc;
use incin::prelude::*;

dim!(Batch, Seq, Feature);

fn main() {
    let _dev = DeviceId::cpu();

    #[allow(clippy::type_complexity)]
    let t1: Tensor<s![Batch, 10]> = Tensor::zeros((32usize, ())).unwrap();
    #[allow(clippy::type_complexity)]
    let _t2: Tensor<s![Batch, 20]> = Tensor::zeros((32usize, ())).unwrap();
    #[allow(clippy::type_complexity)]
    let t3: Tensor<s![Batch, 10]> = Tensor::zeros((32usize, ())).unwrap();

    // Should compile
    let _t4 = t1.add_exact(&t3).unwrap();

    // Should fail with shape mismatch: same dim name (Batch) but a
    // different literal size on the other axis (10 vs 20) — proven as a
    // real compile_fail snapshot at
    // crates/incin-core/tests/compile_fail/named_dim_size_mismatch.rs.
    // let _t5 = t1.add(&_t2).unwrap(); // This correctly fails to compile!
    println!("Compiled successfully!");
}
