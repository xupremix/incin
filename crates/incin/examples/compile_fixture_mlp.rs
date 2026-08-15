use incin::prelude::*;

fn main() -> Result<()> {
    let model = Sequential(
        Linear::<s![8, 16]>::build(())?,
        Sequential(Linear::<s![16, 4]>::build(())?, ReLU),
    );
    let output = model.forward(Tensor::<s![2, 8]>::ones(())?)?;
    assert_eq!(output.dims().dims(), &[2, 4]);
    Ok(())
}
