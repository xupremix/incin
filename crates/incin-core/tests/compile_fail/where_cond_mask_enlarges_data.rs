extern crate incin_core as incin;

use incin::prelude::*;
use incin_backends::cpu::CpuBackendImpl;

fn main() {
    // #100's directional pin refuses a mask that would *enlarge* the data:
    // `[1, 1, 3]` broadcasts against `[2, 3]` to `[1, 2, 3]`, which is not
    // the data's own shape type, so `Output = S2` has no matching impl and
    // the call cannot compile. Broadcast-compatible but wrong-direction.
    let mask = Tensor::<s![1, 1, 3], CpuBackendImpl, bool>::from_slice(&[true; 3], ()).unwrap();
    let on_true = Tensor::<s![2, 3], CpuBackendImpl, f32>::zeros(()).unwrap();
    let on_false = Tensor::<s![2, 3], CpuBackendImpl, f32>::zeros(()).unwrap();

    let _ = mask.where_cond(&on_true, &on_false);
}
