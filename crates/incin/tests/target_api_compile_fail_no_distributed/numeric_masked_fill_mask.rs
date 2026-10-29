use incin::prelude::*;

fn main() -> Result<()> {
    let x = Cpu.tensor([1.0_f32, 2.0])?;
    let mask = Cpu.tensor([1_u8, 0])?;
    let _ = x.masked_fill(&mask, 0.0)?;
    Ok(())
}
