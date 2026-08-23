//! Example: exercises the documented API around `main`.
use incin::prelude::*;

fn main() -> Result<()> {
    let conv = Conv2d::<s![dyn, dyn, 3, 1, 0, 1]>::build((4, 1))?;
    let features = Sequential(conv, ReLU);
    let output = features.forward(Tensor::<Dyn>::zeros(vec![2, 1, 8, 8])?)?;
    assert_eq!(output.dims().dims(), &[2, 4, 6, 6]);
    Ok(())
}
