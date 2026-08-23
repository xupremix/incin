//! Integration coverage for `main` on the documented public surface.
use incin::prelude::*;

fn main() {
    let _ = Cpu.zeros(shape![2i32, 5]);
}
