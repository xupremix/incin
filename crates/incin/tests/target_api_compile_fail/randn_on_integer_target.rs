//! Integration coverage for `main` on the documented public surface.
use incin::prelude::*;

fn main() -> Result<()> {
    // `zeros` on an integer target is fine; sampling a continuous
    // distribution into one is not.
    let idx = Cpu.dtype::<i64>()?;
    let _ = idx.randn(shape![4]);
    Ok(())
}
