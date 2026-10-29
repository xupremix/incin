// Issue #93, Decision 3: `mish` has no quantized kernel path, so a statically
// `Q8_0` operand is refused at the call site naming `FloatCapable`.
extern crate incin_core as incin;

use incin::prelude::*;
use incin_backends::cpu::CpuBackendImpl;

fn main() {
    let t = Tensor::<s![64], CpuBackendImpl>::zeros(()).unwrap();
    let q = t.quantize(-1).unwrap();
    let _ = q.mish();
}
