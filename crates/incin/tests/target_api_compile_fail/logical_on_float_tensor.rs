//! Integration coverage for `main` on the documented public surface.
use incin::prelude::*;

fn main() -> Result<()> {
    let a = Cpu.tensor([1.0_f32, 2.0])?;
    let b = Cpu.tensor([1.0_f32, 0.0])?;
    let _ = a.logical_and(&b);
    Ok(())
}
