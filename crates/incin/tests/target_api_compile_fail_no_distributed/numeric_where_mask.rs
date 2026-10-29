use incin::prelude::*;

fn main() -> Result<()> {
    let mask = Cpu.tensor([1_i64, 0, 1])?;
    let yes = Cpu.tensor([1_i64, 2, 3])?;
    let no = Cpu.tensor([4_i64, 5, 6])?;
    let _ = mask.where_cond(&yes, &no)?;
    Ok(())
}
