use incin::prelude::*;

fn main() -> Result<()> {
    let query = Tensor::<s![4, 8]>::ones(())?;
    let key = Tensor::<s![8, 4]>::ones(())?;
    let value = Tensor::<s![4, 8]>::ones(())?;
    let scores = query.matmul(&key)?;
    let output = scores.softmax(1)?.matmul(&value)?;
    assert_eq!(output.dims().dims(), &[4, 8]);
    Ok(())
}
