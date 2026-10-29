extern crate incin_core as incin;

use incin::prelude::*;
use incin_backends::cpu::CpuBackendImpl;

fn main() {
    // #100's fail-closed direction: a `[2, 4]` mask against `[3, 4]` data is
    // not broadcast-compatible on axis 0 (neither extent is 1), so
    // `where_cond`'s directional pin `S: BroadcastShape<S2, Output = S2>`
    // has no impl to select and the call cannot compile.
    let mask = Tensor::<s![2, 4], CpuBackendImpl, bool>::from_slice(&[false; 8], ()).unwrap();
    let on_true = Tensor::<s![3, 4], CpuBackendImpl, f32>::zeros(()).unwrap();
    let on_false = Tensor::<s![3, 4], CpuBackendImpl, f32>::zeros(()).unwrap();

    let _ = mask.where_cond(&on_true, &on_false);
}
