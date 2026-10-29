//! Compile-pass for #100's acceptance case: a typed rank-2 causal mask
//! selects over `[B, H, T, T]` scores, and the result keeps the scores'
//! own static shape type - no explicit `broadcast_to`, no `Dyn` escape.
extern crate incin_core as incin;

use incin::prelude::*;
use incin_backends::cpu::CpuBackendImpl;

fn main() {
    let scores = Tensor::<s![2, 4, 8, 8], CpuBackendImpl, f32>::zeros(()).unwrap();
    let neg_inf = Tensor::<s![2, 4, 8, 8], CpuBackendImpl, f32>::zeros(()).unwrap();
    let mask = Tensor::<s![8, 8], CpuBackendImpl, bool>::from_slice(&[false; 64], ()).unwrap();

    // The directional pin: `s![8, 8]: BroadcastShape<s![2, 4, 8, 8],
    // Output = s![2, 4, 8, 8]>` holds via rank promotion, and the return
    // type names the data's shape, not the mask's.
    let _masked: Tensor<s![2, 4, 8, 8], CpuBackendImpl, f32, NoGrad, Local, RowMajor> =
        mask.where_cond(&scores, &neg_inf).unwrap();
}
