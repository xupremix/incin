// Issue #93, Decision 3: `dequantize` reads block storage, so a statically
// `f32` operand is refused at the call site naming `QuantCapable`.
extern crate incin_core as incin;

use incin::prelude::*;
use incin_backends::cpu::CpuBackendImpl;

fn main() {
    let t = Tensor::<s![64], CpuBackendImpl>::zeros(()).unwrap();
    let _ = t.dequantize::<f32>();
}
