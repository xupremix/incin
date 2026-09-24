// Issue #93, Decision 4: a statically known last axis that is not a whole
// multiple of the 32-element Q8_0 block fails at monomorphization, before
// any kernel runs. 48 is not divisible by 32.
extern crate incin_core as incin;

use incin::prelude::*;
use incin_backends::cpu::CpuBackendImpl;

fn main() {
    let t = Tensor::<s![48], CpuBackendImpl>::zeros(()).unwrap();
    let _ = t.quantize(-1);
}
