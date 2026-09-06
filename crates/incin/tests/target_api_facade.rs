//! The target-first allocation API as an ordinary user reaches it: through the
//! `incin` façade's prelude, with no `incin_backends` import.
//!
//! The prototype's own tests live in `incin-backends` and import it directly.
//! This one exists because the extension traits only resolve when in scope,
//! so "is it actually reachable from the public façade" is a separate
//! question from "does it work".

#![cfg(feature = "cpu")]

use incin::prelude::*;

#[test]
fn the_whole_surface_is_reachable_from_the_facade_prelude() {
    let cpu = Cpu;

    // Data keeps its own dtype; the target's float is not imposed on it.
    let x = cpu.tensor([[1.0_f32, 2.0], [3.0, 4.0]]).unwrap();
    assert_eq!(x.dims(), [2, 2]);
    let labels = cpu.tensor([0_i64, 1]).unwrap();
    assert_eq!(labels.to_vec1::<i64>().unwrap(), vec![0, 1]);

    // All three shape certainties, via `shape!`.
    let batch = 3usize;
    assert_eq!(cpu.zeros(shape![2, 3]).unwrap().dims(), [2, 3]);
    assert_eq!(cpu.zeros(shape![batch, 4]).unwrap().dims(), [3, 4]);
    assert_eq!(&cpu.zeros([batch, 4]).unwrap().dims()[..], &[3, 4][..]);

    // Layers begin at the canonical target-aware builder.
    let layer = incin_core::nn::linear::linear(shape![4, 3])
        .init(&cpu)
        .unwrap();
    assert_eq!(layer.weight.shape_dims(), vec![3, 4]);
}

/// Guarantees that are compile errors rather than runtime failures.
///
/// These live here rather than as `compile_fail` doctests next to the methods
/// because rustdoc only collects doctests from a crate's *library* target -
/// a `compile_fail` block in an integration test file is never executed and
/// asserts nothing.
#[test]
fn target_api_compile_fail_diagnostics() {
    // Snapshots here are recorded under CI's invocation:
    //
    //   cargo test --all-targets --no-default-features \
    //     --features incin-backends/cpu,incin/cpu
    //
    // `numeric_where_mask` prints the impl's self type in a "the method was
    // found for" note, and that type now includes `Local`, because naming the
    // layout parameter forces the placement before it to be named too. The
    // facade prelude gates `Local` behind the `distributed` feature, which
    // that invocation does not enable, so rustc renders the canonical
    // `incin_core::dist::placement::Local` path here. Feature sets that do
    // enable `distributed` (including plain workspace builds, via feature
    // unification) render the shorter `incin::prelude::Local` instead, and
    // only that spelling can be stored: running this under a different
    // feature set reports a mismatch that is purely the path spelling.
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/target_api_compile_fail/*.rs");
    // A second directory rather than a `cfg` inside a fixture: trybuild
    // compiles each fixture as a whole file, and a body that `cfg`s itself out
    // compiles clean, which a `compile_fail` case reports as a failure.
    #[cfg(feature = "wgpu")]
    t.compile_fail("tests/target_api_compile_fail_wgpu/*.rs");
}
