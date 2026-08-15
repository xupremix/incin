use incin::prelude::*;

fn main() -> Result<()> {
    let input = Tensor::<s![2, 4]>::ones(())?;
    let weight = Tensor::<s![4, 3]>::ones(())?;
    let output = input.matmul(&weight)?;
    assert_eq!(output.dims().dims(), &[2, 3]);
    Ok(())
}
