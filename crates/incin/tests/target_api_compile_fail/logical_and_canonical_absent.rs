//! Integration coverage for `main` on the documented public surface.
use incin::prelude::*;

fn main() -> Result<()> {
    let a = Cpu.dtype::<bool>()?.ones([2])?;
    let b = Cpu.dtype::<bool>()?.ones([2])?;
    let _ = a.logical_and_canonical(&b);
    Ok(())
}
