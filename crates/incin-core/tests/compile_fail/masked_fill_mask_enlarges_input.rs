extern crate incin_core as incin;

use incin::prelude::*;
use incin_backends::cpu::CpuBackendImpl;

fn main() {
    // The `masked_fill` mirror of #100's pin: the mask broadcasts *into* the
    // input, so `S: BroadcastShape<S2, Output = S>` requires the broadcast
    // to resolve to the input's own shape. `[1, 1, 3]` against `[2, 3]`
    // broadcasts to `[1, 2, 3]`, which is not `[2, 3]`, so there is no
    // matching `Output` impl and the call cannot compile.
    let x = Tensor::<s![2, 3], CpuBackendImpl, f32>::zeros(()).unwrap();
    let mask = Tensor::<s![1, 1, 3], CpuBackendImpl, bool>::from_slice(&[true; 3], ()).unwrap();

    let _ = x.masked_fill(&mask, 0.0);
}
