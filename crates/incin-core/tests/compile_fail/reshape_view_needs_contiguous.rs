//! `reshape_view` reinterprets elements without copying, which is correct only
//! when they form one unbroken run in shape order. A channels-last buffer's do
//! not, so the bound must reject it.
//!
//! Before `ChannelsLast` existed every layout in the crate implemented
//! `Contiguous`, so this bound had never rejected anything: it was vacuously
//! satisfied by the whole inhabited world. This case is what makes it a check.
//!
//! Written against a parameter rather than a constructed value on purpose. A
//! channels-last tensor cannot be *built* yet -- `zeros` is bounded on the
//! sealed `FreshDense`, and the target API refuses a layout whose strides no
//! backend can allocate -- so constructing one here would fail for a second,
//! unrelated reason and the case would stop pinning one thing.
extern crate incin_core as incin;
use incin_backends::cpu::CpuBackendImpl;
use incin_core::prelude::*;
use incin_core::shapes::ChannelsLast;
use incin_macros::s;

type Nchw = s![2, 3, 4, 5];

fn flatten(
    t: Tensor<Nchw, CpuBackendImpl, f32, NoGrad, Local, ChannelsLast>,
) -> Tensor<s![120], CpuBackendImpl, f32, NoGrad, Local, RowMajor> {
    t.reshape_view::<s![120]>().unwrap()
}

fn main() {
    let _ = flatten;
}
