//! #149: the inference-loop release valve.
//!
//! A forward that is never followed by `backward` retains its graph for the
//! life of the thread — an evaluation pass that poses forwards and walks
//! only some of them grows without bound. `GradMode::Disabled.scope` avoids
//! recording in the first place, but a graph recorded before its consumer
//! decided against `backward` still needs a way out. That way out is
//! [`drain_tape`](incin_backends::cpu::drain_tape): discard every
//! unevaluated node without walking, paired with
//! [`tape_depth`](incin_backends::cpu::tape_depth) as the gauge.
//!
//! Each test runs on its own thread, so the thread-local tape starts empty;
//! the tests drain first anyway, because a guarantee that depends on thread
//! reuse never happening is not a guarantee.

extern crate incin_core as incin;

use incin_backends::cpu::{CpuBackendImpl, drain_tape, tape_depth};
use incin_core::prelude::*;
use incin_macros::s;

type B = CpuBackendImpl;

fn record_one_graph() {
    let a = Tensor::<s![2, 2], B, f32, Grad>::ones(()).unwrap();
    let b = Tensor::<s![2, 2], B, f32, Grad>::ones(()).unwrap();
    let _loss = a.matmul(&b).unwrap().sum_all().unwrap();
}

#[test]
fn drain_discards_recorded_graphs_without_walking() {
    drain_tape();
    record_one_graph();
    assert!(tape_depth() > 0, "the matmul graph should be on the tape");
    drain_tape();
    assert_eq!(tape_depth(), 0, "draining must empty the tape");
}

#[test]
fn drain_is_a_noop_when_nothing_was_recorded() {
    drain_tape();
    assert_eq!(tape_depth(), 0);
    drain_tape();
    assert_eq!(tape_depth(), 0, "draining an empty tape must stay empty");
}

#[test]
fn disabled_scope_still_records_nothing() {
    drain_tape();
    incin_core::exec::GradMode::Disabled.scope(|| {
        record_one_graph();
    });
    assert_eq!(
        tape_depth(),
        0,
        "a disabled scope records nothing, so there is nothing to drain"
    );
}
