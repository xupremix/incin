//! Minimal smoke test for `symbolic_dim!` — for a fuller tour (transpose and
//! concat preserving names, a realistic name-mismatch, the exact compiler
//! error it produces), see `crates/incin/examples/named_dims_safety`.
extern crate alloc;
use incin::prelude::*;

symbolic_dim!(Batch, Seq, Feature);

fn main() {
    let _dev = DeviceId::cpu();

    let t1: Tensor<s![Batch, 10]> = Tensor::zeros((32usize, ())).unwrap();
    let _t2: Tensor<s![Batch, 20]> = Tensor::zeros((32usize, ())).unwrap();
    let t3: Tensor<s![Batch, 10]> = Tensor::zeros((32usize, ())).unwrap();

    // Should compile
    let _t4 = t1.add(&t3).unwrap();

    // Should fail with shape mismatch: same dim name (Batch) but a
    // different literal size on the other axis (10 vs 20) — proven as a
    // real compile_fail snapshot at
    // crates/incin-core/tests/compile_fail/named_dim_size_mismatch.rs.
    // let _t5 = t1.add(&_t2).unwrap(); // This correctly fails to compile!
    println!("Compiled successfully!");
}
