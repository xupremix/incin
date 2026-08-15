#![allow(clippy::type_complexity)]

extern crate alloc;
use incin::prelude::*;

fn main() -> Result<()> {
    println!("--- Static MatMul Example ---");

    // We instantiate two tensors with static dimensions.
    // t1 is 3x4
    let t1: Tensor<s![3, 4]> = Tensor::zeros(())?;

    // t2 is 4x5
    let t2: Tensor<s![4, 5]> = Tensor::zeros(())?;

    println!("t1 shape: {:?}", t1.dims());
    println!("t2 shape: {:?}", t2.dims());

    // Perform matrix multiplication!
    // The compiler strictly verifies that `4 == 4` (the inner dimensions match).
    let t3 = t1.matmul(&t2)?;

    println!("t3 shape after matmul: {:?}", t3.dims());
    println!("If you changed t2 to 5x5 in the code, `cargo check` would fail instantly!");

    Ok(())
}
