#![allow(clippy::type_complexity)]

extern crate alloc;
use incin::prelude::*;

fn main() -> Result<()> {
    println!("--- Static MatMul Example ---");

    // We instantiate two tensors with static dimensions using the Target API.
    // t1 is 3x4 (Gaussian random values)
    let t1 = Cpu.randn(shape![3, 4])?;

    // t2 is 4x5 (Gaussian random values)
    let t2 = Cpu.randn(shape![4, 5])?;

    println!("t1 shape: {:?}", t1.dims());
    println!("t2 shape: {:?}", t2.dims());

    // Perform matrix multiplication!
    // The compiler strictly verifies that `4 == 4` (the inner dimensions match).
    let t3 = t1.matmul(&t2)?;

    println!("t3 shape after matmul: {:?}", t3.dims());
    println!("t3 result sample:\n{t3}");
    println!(
        "If you changed t2 to shape![5, 5] in the code, `cargo check` would fail instantly at compile time!"
    );

    Ok(())
}
