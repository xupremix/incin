//! Verifies `ComputeStats` (docs/growth/04-compile-time-stats.md, v1) against
//! hand-computed numbers — including the exact params/MACs example numbers
//! (101,770 / 101,632) from the growth doc's own pretty-print mockup, so
//! this test doubles as a check that the doc's illustrative numbers and the
//! real implementation actually agree.
extern crate kindle_core as kindle;
use kindle_core::prelude::dummy::DummyBackend;
use kindle_core::prelude::*;

type TestBackend = DummyBackend<f32, Cpu>;

#[module]
struct Mlp<B: Backend> {
    fc1: Linear<s![784, 128], B>,
    fc2: Linear<s![128, 10], B>,
}

#[test]
fn linear_stats_match_hand_computed_formula() {
    let layer = Linear::<s![784, 128], TestBackend>::build(()).unwrap();
    let stats = layer.compute_stats(1);
    assert_eq!(stats.params, 784 * 128 + 128); // weight + bias
    assert_eq!(stats.macs, 784 * 128);

    let stats_batch_32 = layer.compute_stats(32);
    assert_eq!(stats_batch_32.params, stats.params); // params don't scale with batch
    assert_eq!(stats_batch_32.macs, 784 * 128 * 32);
}

#[test]
fn mlp_stats_match_the_growth_docs_own_worked_example() {
    let model = Mlp::<TestBackend> {
        fc1: Linear::build(()).unwrap(),
        fc2: Linear::build(()).unwrap(),
    };

    let stats = model.compute_stats(1);
    // docs/growth/04-compile-time-stats.md's pretty-print mockup gives these
    // exact totals for this exact Mlp shape — this test is also a check
    // that the doc's own illustrative numbers are correct, not just that
    // this implementation is internally consistent.
    assert_eq!(stats.params, 101_770);
    assert_eq!(stats.macs, 101_632);

    let model_stats = stats.into_model_stats();
    assert_eq!(model_stats.params, 101_770);
    assert_eq!(model_stats.macs, 101_632);
    assert_eq!(model_stats.flops, 101_632 * 2);
}

#[test]
fn mlp_stats_scale_macs_with_batch_but_not_params() {
    let model = Mlp::<TestBackend> {
        fc1: Linear::build(()).unwrap(),
        fc2: Linear::build(()).unwrap(),
    };

    let stats = model.compute_stats(8);
    assert_eq!(stats.params, 101_770);
    assert_eq!(stats.macs, 101_632 * 8);
}

#[test]
fn sequential_with_an_activation_sums_correctly_and_the_activation_contributes_nothing() {
    // Proves the container path (Sequential<L1, L2>) and the zero-compute
    // activation impls (ReLU here) both work, and compose correctly.
    let seq = Sequential(Linear::<s![4, 8], TestBackend>::build(()).unwrap(), ReLU);
    let stats = seq.compute_stats(2);
    assert_eq!(stats.params, 4 * 8 + 8);
    assert_eq!(stats.macs, 4 * 8 * 2);
}

#[test]
fn conv2d_params_are_exact_but_macs_are_the_documented_v1_gap() {
    // Conv2d's params come from the same generic "sum of Param fields"
    // path every #[module] struct gets automatically (no hand-written
    // formula needed) — so this is exact. Its MACs need the input's
    // spatial size, which Conv2d doesn't store, so v1 honestly reports 0
    // rather than guessing; see nn/stats.rs's module docs.
    let conv = Conv2d::<s![16, 3, 3, 1, 1, 1], TestBackend>::build(()).unwrap();
    let stats = conv.compute_stats(1);
    assert_eq!(stats.params, 16 * 3 * 3 * 3 + 16); // weight (16,3,3,3) + bias (16,)
    assert_eq!(stats.macs, 0);
}

#[test]
fn summary_with_stats_appends_a_readable_totals_footer() {
    let model = Mlp::<TestBackend> {
        fc1: Linear::build(()).unwrap(),
        fc2: Linear::build(()).unwrap(),
    };

    let out = model.summary_with_stats(1);
    assert!(out.contains("Total params: 101770"));
    assert!(out.contains("MACs: 101632"));
    assert!(out.contains("FLOPs: 203264"));
    // Still a superset of the plain summary (extends by composition, per
    // format_layer_summary_with_stats's own doc comment).
    assert!(out.contains(&model.summary()));
}
