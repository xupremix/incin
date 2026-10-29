//! End-to-end `MultiHeadAttention` on a real WGPU adapter, against the CPU
//! twin with the same weights (issue #91: "an attention block runs end to
//! end on WGPU").
//!
//! Weight initialization is backend-local (CPU samples `thread_rng`, WGPU a
//! time-seeded LCG), so the WGPU module is built, then loaded from the CPU
//! module's [`collect_state`] snapshot before either forward runs. Every
//! comparison is the same math on the same numbers, only the device differs.
//!
//! Covers the paths attention actually takes: fused SDPA without a mask,
//! the causal mask (`ones` + `tril` + `log` + fused SDPA with mask),
//! grouped-query head expansion, and rotary table construction — plus a
//! training smoke that walks the tape backward on WGPU alone.
//!
//! Dropout with `p > 0` is intentionally absent from the twin comparisons:
//! both backends seed their keep-mask from the clock, so the outputs cannot
//! match by construction. A separate test only asserts that the composed
//! path runs and stays finite on WGPU.
#![cfg(all(feature = "wgpu", feature = "cpu"))]

use incin_backends::cpu::CpuBackendImpl;
use incin_backends::wgpu::{WgpuBackendImpl, tape_depth};
use incin_core::backend_authoring::HostInterop;
use incin_core::nn::{AttentionConfig, MultiHeadAttention, collect_state, load_state};
use incin_core::prelude::{DTypeId, DeviceId, Dyn, Module, NoGrad, Tensor, WgpuN};
use incin_core::typenum::U0;

type Cpu = CpuBackendImpl;
type Wgpu = WgpuBackendImpl<WgpuN<U0>>;

/// Aborts unless a WGPU adapter can allocate — same contract as the other
/// WGPU suites: enabling the feature is an explicit request for the backend.
fn require_wgpu() {
    assert!(
        <Wgpu as HostInterop>::from_bytes::<f32>(
            &[0u8; 4],
            &[1],
            DTypeId::F32.descriptor(),
            &DeviceId::wgpu(0),
        )
        .is_ok(),
        "no WGPU adapter, but the `wgpu` feature is enabled"
    );
}

/// Deterministic, non-symmetric input — same recipe as the CPU attention
/// suite, so a transposed score matrix cannot compare equal to itself.
fn ramp_cpu(dims: Vec<usize>) -> Tensor<Dyn, Cpu, f32, NoGrad> {
    let n: usize = dims.iter().product();
    let values: Vec<f32> = (0..n)
        .map(|v| ((v as f32) * 0.37).sin() * 0.9 + (v as f32) * 0.01)
        .collect();
    Tensor::<Dyn, Cpu>::from_slice(&values, dims).expect("cpu ramp uploads")
}

fn ramp_wgpu(dims: Vec<usize>) -> Tensor<Dyn, Wgpu, f32, NoGrad> {
    let n: usize = dims.iter().product();
    let values: Vec<f32> = (0..n)
        .map(|v| ((v as f32) * 0.37).sin() * 0.9 + (v as f32) * 0.01)
        .collect();
    Tensor::<Dyn, Wgpu>::from_slice(&values, dims).expect("wgpu ramp uploads")
}

fn assert_close(actual: &[f32], expected: &[f32], tol: f32, label: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{label}: length mismatch ({} vs {})",
        actual.len(),
        expected.len()
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{label}[{i}]: got {a}, expected {e} (|Δ|={}; tol {tol})",
            (a - e).abs()
        );
    }
}

/// Builds the same attention on CPU and WGPU with identical weights, runs
/// both forwards on the same ramp input, and returns `(cpu, wgpu)`.
fn twin_forward<const D_MODEL: usize, const N_HEADS: usize, const N_KV_HEADS: usize>(
    config: AttentionConfig,
    dims: Vec<usize>,
) -> (Vec<f32>, Vec<f32>) {
    let cpu = MultiHeadAttention::<D_MODEL, N_HEADS, N_KV_HEADS, Cpu>::build(config, (), ())
        .expect("cpu attention builds");
    let snapshot = collect_state::<Cpu, _>(&cpu).expect("cpu state collects");

    let mut gpu = MultiHeadAttention::<D_MODEL, N_HEADS, N_KV_HEADS, Wgpu>::build(config, (), ())
        .expect("wgpu attention builds");
    load_state::<Wgpu, _>(&mut gpu, &snapshot).expect("weights load onto wgpu");
    let round_trip = collect_state::<Wgpu, _>(&gpu).expect("wgpu state collects");
    assert_eq!(
        round_trip, snapshot,
        "the wgpu module did not keep the cpu snapshot"
    );

    let cpu_out = cpu
        .forward(ramp_cpu(dims.clone()))
        .expect("cpu forward")
        .to_vec1::<f32>()
        .expect("cpu readback");
    let gpu_out = gpu
        .forward(ramp_wgpu(dims))
        .expect("wgpu forward")
        .to_vec1::<f32>()
        .expect("wgpu readback");
    (cpu_out, gpu_out)
}

/// Non-causal, zero dropout: the fused SDPA row without a mask.
#[test]
fn non_causal_attention_runs_end_to_end_and_matches_cpu() {
    require_wgpu();
    let (cpu, gpu) = twin_forward::<8, 2, 2>(AttentionConfig::default(), vec![2, 5, 8]);
    assert_eq!(gpu.len(), 2 * 5 * 8, "forward keeps the input shape");
    assert!(
        gpu.iter().all(|v| v.is_finite()),
        "wgpu attention produced a non-finite value"
    );
    assert_close(&gpu, &cpu, 1e-4, "non-causal attention");
}

/// Causal decoder path: `ones` + `tril` + `log` builds the additive mask,
/// then fused SDPA consumes it — the rows #91 names for attention reachability.
#[test]
fn causal_attention_runs_end_to_end_and_matches_cpu() {
    require_wgpu();
    let (cpu, gpu) = twin_forward::<8, 2, 2>(AttentionConfig::causal(), vec![2, 6, 8]);
    assert_eq!(gpu.len(), 2 * 6 * 8);
    assert!(
        gpu.iter().all(|v| v.is_finite()),
        "causal forward non-finite"
    );
    assert_close(&gpu, &cpu, 1e-4, "causal attention");
}

/// Grouped-query attention: `expand_kv_heads` (unsqueeze + broadcast +
/// reshape) runs when `N_KV_HEADS < N_HEADS`.
#[test]
fn grouped_query_attention_runs_end_to_end_and_matches_cpu() {
    require_wgpu();
    let (cpu, gpu) = twin_forward::<8, 4, 2>(AttentionConfig::default(), vec![1, 4, 8]);
    assert_eq!(gpu.len(), 32);
    assert!(gpu.iter().all(|v| v.is_finite()), "gqa forward non-finite");
    assert_close(&gpu, &cpu, 1e-4, "grouped-query attention");
}

/// Rotary positions: building the module constructs the cos/sin tables on
/// WGPU (`arange`/`exp`/`sin`/`cos`), and the forward rotates through them.
#[test]
fn rotary_causal_attention_runs_end_to_end_and_matches_cpu() {
    require_wgpu();
    let config = AttentionConfig::causal().with_rotary(10_000.0, 32);
    let (cpu, gpu) = twin_forward::<8, 2, 2>(config, vec![2, 5, 8]);
    assert_eq!(gpu.len(), 2 * 5 * 8);
    assert!(
        gpu.iter().all(|v| v.is_finite()),
        "rotary forward non-finite"
    );
    assert_close(&gpu, &cpu, 1e-4, "rotary causal attention");
}

/// Editing a later token must not move earlier outputs — the causal mask
/// property, checked on the device that owns the mask rows.
#[test]
fn causal_mask_does_not_leak_on_wgpu() {
    require_wgpu();
    let seq = 5;
    let width = 8;
    let config = AttentionConfig::causal();
    let cpu = MultiHeadAttention::<8, 2, 2, Cpu>::build(config, (), ()).expect("cpu builds");
    let snapshot = collect_state::<Cpu, _>(&cpu).expect("cpu state");
    let mut gpu = MultiHeadAttention::<8, 2, 2, Wgpu>::build(config, (), ()).expect("wgpu builds");
    load_state::<Wgpu, _>(&mut gpu, &snapshot).expect("load");

    let base = ramp_wgpu(vec![1, seq, width]);
    let mut disturbed_values = base.to_vec1::<f32>().expect("read base");
    for value in disturbed_values.iter_mut().skip((seq - 1) * width) {
        *value += 3.5;
    }
    let disturbed =
        Tensor::<Dyn, Wgpu>::from_slice(&disturbed_values, vec![1, seq, width]).expect("upload");

    let before = gpu
        .forward(base)
        .expect("forward base")
        .to_vec1::<f32>()
        .expect("read base out");
    let after = gpu
        .forward(disturbed)
        .expect("forward disturbed")
        .to_vec1::<f32>()
        .expect("read disturbed out");

    let prefix = (seq - 1) * width;
    let prefix_diff = before[..prefix]
        .iter()
        .zip(&after[..prefix])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        prefix_diff < 1e-5,
        "editing the last token moved earlier outputs by {prefix_diff:e}; the mask leaks"
    );
    let last_diff = before[prefix..]
        .iter()
        .zip(&after[prefix..])
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        last_diff > 1e-4,
        "the last output ignored its own token; the mask masks too much ({last_diff:e})"
    );
}

/// Training smoke: a causal forward records a tape, and `backward` walks it
/// to finite, non-zero gradients on every projection weight.
#[test]
fn attention_trains_on_wgpu_with_finite_nonzero_weight_gradients() {
    require_wgpu();
    let width = 8;
    let attention = MultiHeadAttention::<8, 2, 2, Wgpu>::build(
        AttentionConfig::causal().with_rotary(10_000.0, 32),
        (),
        (),
    )
    .expect("wgpu attention builds");
    let x = ramp_wgpu(vec![2, 4, width]).require_grad();
    let target = Tensor::<Dyn, Wgpu>::zeros(vec![2, 4, width]).expect("zeros");

    let before = tape_depth();
    let output = attention.forward(x).expect("forward");
    assert!(
        output
            .to_vec1::<f32>()
            .expect("read")
            .iter()
            .all(|v| v.is_finite()),
        "attention produced a non-finite value"
    );
    assert!(
        tape_depth() > before,
        "a trainable wgpu forward recorded no tape entries"
    );

    let loss = output.mse_loss(&target).expect("mse");
    let grads = loss.backward().expect("backward on wgpu");

    let mut moved = 0usize;
    for (name, parameter) in [
        ("query", &attention.query),
        ("key", &attention.key),
        ("value", &attention.value),
        ("output", &attention.output),
    ] {
        let weight = parameter.weight.as_tensor().expect("weight storage");
        let gradient = grads
            .require(&weight)
            .unwrap_or_else(|e| panic!("no gradient reached {name}.weight: {e}"));
        let values = gradient.to_vec1::<f32>().expect("grad read");
        assert!(
            values.iter().all(|v| v.is_finite()),
            "{name}.weight received a non-finite gradient"
        );
        moved += values.iter().filter(|v| **v != 0.0).count();
    }
    assert!(moved > 0, "every projection gradient was exactly zero");
}

/// The composed training path with attention-weight dropout: non-deterministic
/// across backends (clock-seeded keep-mask), so this only proves it runs and
/// stays finite on WGPU — the fail-closed bar for "reachable", not equality.
#[test]
fn composed_dropout_training_path_runs_on_wgpu() {
    require_wgpu();
    let config = AttentionConfig::causal().with_dropout(0.5);
    let attention = MultiHeadAttention::<8, 2, 2, Wgpu>::build(config, (), ()).expect("build");
    // Dropout starts in training mode; p > 0 selects the composed path.
    assert!(attention.dropout.is_training && attention.dropout.p > 0.0);
    let x = ramp_wgpu(vec![2, 4, 8]);
    let output = attention.forward(x).expect("composed forward");
    let values = output.to_vec1::<f32>().expect("read");
    assert!(
        values.iter().all(|v| v.is_finite()),
        "dropout attention produced a non-finite value"
    );
}

/// Size-1 batch dims survive WGPU matmul — the regression behind the two
/// batch=1 attention failures (CPU twin:
/// `batched_matmul_size_one_batch_is_broadcast_not_unwrapped`).
#[test]
fn size_one_batch_matmul_keeps_the_leading_one() {
    require_wgpu();
    // [1,5,8] x [8,5] -> [1,5,5], not [5,5].
    let x = ramp_wgpu(vec![1, 5, 8]);
    let w = ramp_wgpu(vec![8, 5]);
    let m = x.matmul(&w).expect("size-1-batch matmul");
    assert_eq!(m.dims(), [1, 5, 5], "size-1 batch must not unwrap");
    assert_eq!(
        m.to_vec1::<f32>().expect("read").len(),
        5 * 5,
        "output carries the broadcast batch"
    );

    // [1,3,4] x [5,4,6] -> [5,3,6]: the size-1 side broadcasts, the
    // batched side's leading dim is the output's.
    let lhs = ramp_wgpu(vec![1, 3, 4]);
    let rhs = ramp_wgpu(vec![5, 4, 6]);
    let out = lhs.matmul(&rhs).expect("broadcasting matmul");
    assert_eq!(out.dims(), [5, 3, 6]);
}
