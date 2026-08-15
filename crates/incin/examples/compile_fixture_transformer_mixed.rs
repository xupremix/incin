use incin::prelude::*;

fn main() -> Result<()> {
    let projection = Linear::<s![8, 8]>::build(())?;
    let tokens = Tensor::<s![dyn, 8]>::zeros((4, ()))?;
    let output = projection.forward(tokens)?;
    assert_eq!(output.dims().dims(), &[4, 8]);
    Ok(())
}
