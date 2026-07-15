#[macro_use]
extern crate alloc;
use kindle::prelude::*;

fn main() -> Result<()> {
    println!("--- Static MatMul Example ---");

    // We instantiate two tensors with static dimensions.
    // t1 is 3x4
    let t1: Tensor<s![dyn, 3, 4]> = Tensor::<Dyn>::zeros([3, 4])?.into_shape()?;

    // t2 is 4x5
    let t2: Tensor<s![dyn, 4, 5]> = Tensor::<Dyn>::zeros([4, 5])?.into_shape()?;

    println!("t1 shape: {:?}", t1.dims());
    println!("t2 shape: {:?}", t2.dims());

    // Perform matrix multiplication!
    // The compiler strictly verifies that `4 == 4` (the inner dimensions match).
    let t3 = t1.matmul(&t2)?;

    println!("t3 shape after matmul: {:?}", t3.dims());
    println!("If you changed t2 to 5x5 in the code, `cargo check` would fail instantly!");

    Ok(())
}
