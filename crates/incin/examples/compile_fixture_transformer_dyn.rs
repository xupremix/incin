use incin::prelude::*;

fn main() -> Result<()> {
    let projection = Linear::<Dyn>::build((8, 8))?;
    let tokens = Tensor::<Dyn>::zeros(vec![4, 8])?;
    let output = projection.forward(tokens)?;
    assert_eq!(output.dims().dims(), &[4, 8]);
    Ok(())
}
