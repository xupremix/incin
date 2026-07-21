//! Definition-of-done example (NATBACK-11): trains a small CNN classifier
//! (conv2d -> batch_norm -> relu -> max_pool2d -> flatten -> linear) end-to-end
//! (forward -> cross_entropy_loss -> backward -> optimizer step) for multiple
//! epochs on `NativeBackend<f32, Cpu>`, then re-runs the identical model/data
//! on `CandleBackend<f32, Cpu>`, printing both backends' per-epoch loss curves
//! and wall-clock timing side-by-side.
//!
//! This is the one artifact in Phase 5 that exercises the full typed
//! `Tensor<S,B,K,D,G>` / `Module` / `Optimizer` API end-to-end, not raw
//! `Backend::Storage` calls.
#[macro_use]
extern crate alloc;

use kindle::SGD;
use kindle::prelude::*;
use kindle::prelude::{CrossEntropyLoss, Mean, StateDict};
use kindle_backends::candle::CandleBackend;
use kindle_native::NativeBackend;
use kindle_telemetry::reporter::Reporter;

/// Auto-generated documentation for NB.
type NB = NativeBackend<f32, Cpu>;
/// Auto-generated documentation for CB.
type CB = CandleBackend<f32, Cpu>;

// ── Model ────────────────────────────────────────────────────────────────────

/// A small CNN classifier: conv2d -> batch_norm -> relu -> max_pool2d ->
/// flatten -> linear (per D-04). No skip connection -- simpler than
/// `native_resnet.rs`'s `BasicBlock`, purpose-built for this demo.
#[module]
pub struct SimpleCnn<B: Backend> {
    // Bias = False: `Conv2dShape::build_args`'s bias-arg construction
    // (`crates/kindle-core/src/nn/conv2d.rs`, out of this plan's
    // files_modified scope) has a pre-existing bug producing a zero-length
    // bias buffer for `usize`-typed `OutC` (uses `Default::default()`
    // instead of the actual out-channel count), which panics at conv2d
    // forward time. BatchNorm2d immediately follows conv1 and already
    // applies its own learned affine shift, so a conv bias is redundant
    // here regardless -- disabling it sidesteps the bug with no loss of
    // model expressiveness.
    /// Auto-generated documentation for conv1.
    pub conv1: kindle_core::prelude::Conv2d<
        s![dyn, dyn, 3, 1, 1, 1],
        B,
        kindle_core::prelude::False,
    >,
    /// Auto-generated documentation for bn1.
    pub bn1: kindle::BatchNorm2d<s![dyn], B>,
    // MaxPool2d is a zero-sized, stateless module (no parameters, no
    // device-bound buffers) and does not implement `ToDevice` -- it is
    // excluded from the `#[module]`-derived Parameters/StateDict/ToDevice
    // impls, which is semantically correct since it has nothing to collect
    // or move across devices.
    #[module(ignore)]
    /// Auto-generated documentation for pool.
    pub pool: kindle::MaxPool2d<typenum::U2, typenum::U2>,
    /// Auto-generated documentation for fc.
    pub fc: kindle::Linear<Dyn, B>,
}

impl<B: Backend + kindle::ModuleOps<B>> SimpleCnn<B>
where
    B::FloatElem: ConstDType,
    B::Device: ConstDevice,
{
    /// Auto-generated documentation for new.
    pub fn new(
        in_channels: usize,
        conv_out_channels: usize,
        num_classes: usize,
        flattened_dim: usize,
    ) -> Result<Self> {
        Ok(Self {
            conv1: kindle_core::prelude::Conv2d::<
                s![dyn, dyn, 3, 1, 1, 1],
                B,
                kindle_core::prelude::False,
            >::new_with((conv_out_channels, in_channels))?,
            bn1: kindle::BatchNorm2d::<s![dyn], B>::new_with((conv_out_channels,), 1e-5, 0.1)?,
            pool: kindle::MaxPool2d::<typenum::U2, typenum::U2>::new()?,
            fc: kindle::Linear::<Dyn, B>::new_with((flattened_dim, num_classes))?,
        })
    }
}

impl<B: Backend + kindle::ModuleOps<B>> SimpleCnn<B> {
    /// Auto-generated documentation for forward.
    pub fn forward(&self, x: Tensor<Dyn, B>) -> Result<Tensor<Dyn, B>> {
        let x = self.conv1.forward(x)?;
        let x = self.bn1.forward(x)?;
        let x = x.relu()?;
        let x = self.pool.forward(x)?;

        let dims = x.dims();
        let (b, c, h, w) = (dims[0], dims[1], dims[2], dims[3]);
        let x = x.try_reshape::<Dyn>(vec![b, c * h * w])?;

        self.fc.forward(x)
    }
}

// ── Synthetic dataset ────────────────────────────────────────────────────────

/// Builds a small, fully deterministic synthetic dataset: `n_samples` images
/// of shape `[1, 8, 8]` and a 2-class integer label per image. No RNG of any
/// kind is used (neither `rand` nor either backend's own) -- images are
/// generated via a fixed arithmetic pattern seeded only by the sample index,
/// and the label is derived from a hand-constructed separability rule (mean
/// pixel value above/below a fixed threshold), so the loss is genuinely
/// learnable rather than degenerate. Per T-05-06, this is built ONCE and the
/// same bytes are fed to both backends -- never regenerated per-backend.
fn make_dataset() -> (Vec<f32>, Vec<u32>, usize) {
    /// Auto-generated documentation for N.
    const N: usize = 32;
    /// Auto-generated documentation for HW.
    const HW: usize = 8;
    /// Auto-generated documentation for PIXELS.
    const PIXELS: usize = HW * HW;

    let mut images = Vec::with_capacity(N * PIXELS);
    let mut labels = Vec::with_capacity(N);

    for i in 0..N {
        // Deterministic per-sample "brightness" level in [0, 1), derived
        // purely from the sample index (no RNG).
        let level = (i as f32) / (N as f32);
        // Class 0: mostly dark pixels with a small deterministic per-pixel
        // ripple. Class 1: mostly bright pixels with the same ripple.
        // Alternate class assignment by index parity so the dataset is
        // balanced and the mean-pixel-value threshold rule cleanly
        // separates the two classes.
        let label: u32 = (i % 2) as u32;
        let base = if label == 0 { 0.1 } else { 0.9 };

        for p in 0..PIXELS {
            // Small deterministic ripple so pixels within an image aren't
            // all identical, without introducing any randomness.
            let ripple = 0.05 * ((p as f32 * 0.37 + level).sin());
            images.push((base + ripple).clamp(0.0, 1.0));
        }
        labels.push(label);
    }

    (images, labels, N)
}

/// Auto-generated documentation for as_f32_bytes.
fn as_f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_ne_bytes()).collect()
}

/// `NativeBackend::from_bytes` only supports the F32 dtype (confirmed by
/// 05-RESEARCH.md's trait-closure audit -- this is an existing, documented
/// gap, not something this plan's scope covers fixing), and
/// `CandleBackend::from_bytes` always interprets its input bytes as `f32`
/// first regardless of the target dtype (per `native_parity.rs`'s own
/// `cross_entropy_loss_forward_and_backward_parity` precedent comment).
/// Neither backend can build a `u32` label storage through the shared
/// `Backend::from_bytes` trait method from the SAME byte encoding, so this
/// trait supplies each backend's own idiomatic label-construction path from
/// the SAME `&[u32]` class-index values (guaranteeing identical label
/// values across backends, matching T-05-06's intent, even though the two
/// backends' construction mechanisms necessarily differ).
trait MakeLabels: Backend<FloatElem = f32> {
    /// Auto-generated documentation for make_labels.
    fn make_labels(values: &[u32]) -> Self::Storage<u32>;
}

impl MakeLabels for NB {
    /// Auto-generated documentation for make_labels.
    fn make_labels(values: &[u32]) -> Self::Storage<u32> {
        kindle_native::storage::NativeStorage::from_contiguous(
            kindle_native::storage::NativeBuffer::U32(values.to_vec()),
            vec![values.len()],
        )
    }
}

impl MakeLabels for CB {
    /// Auto-generated documentation for make_labels.
    fn make_labels(values: &[u32]) -> Self::Storage<u32> {
        // CandleBackend::from_bytes always reads its input bytes as f32
        // first, then casts to the target dtype -- so label values must be
        // f32-encoded bytes here, not raw u32 bytes.
        let f32_values: Vec<f32> = values.iter().map(|&v| v as f32).collect();
        let bytes = as_f32_bytes(&f32_values);
        Self::from_bytes::<u32>(
            &bytes,
            &[values.len()],
            KindleDType::U32,
            &KindleDevice::cpu(),
        )
        .expect("from_bytes labels (Candle)")
    }
}

// ── Training loop ────────────────────────────────────────────────────────────

/// Trains `SimpleCnn<B>` end-to-end (forward -> cross_entropy_loss ->
/// backward -> optimizer step) for `n_epochs` epochs on the given synthetic
/// dataset bytes, returning the per-epoch loss curve and total wall-clock
/// duration.
///
/// Optimizer choice: SGD (Claude's Discretion per CONTEXT.md D-04). SGD
/// produced a monotonically decreasing loss curve on this task's simple
/// mean-pixel-value separability rule without needing Adam's adaptive
/// per-parameter learning rates -- a plain fixed learning rate is enough for
/// a 2-class linearly-separable-after-conv problem this small.
///
/// `clippy::too_many_arguments` is allowed here: this is example/test-harness
/// code (not library `src/`), and the extra `reporter`/`grad_sample_every_n`/
/// `init_state` parameters are this plan's (07-04) telemetry-dogfooding
/// extension of an otherwise-unchanged existing function -- bundling them
/// into a config struct would be pure ceremony for a single call site.
#[allow(clippy::too_many_arguments)]
fn train<B: MakeLabels + kindle::LossOps<B> + kindle::ModuleOps<B>>(
    images_bytes: &[u8],
    label_values: &[u32],
    n_samples: usize,
    n_epochs: usize,
    lr: f64,
    reporter: Option<&kindle_telemetry::emitter::Emitter>,
    grad_sample_every_n: Option<usize>,
    init_state: Option<&BTreeMap<String, Tensor<Dyn, B>>>,
) -> (Vec<f64>, std::time::Duration, SimpleCnn<B>)
where
    B::FloatElem: ConstDType,
    B::Device: ConstDevice,
{
    let device = KindleDevice::cpu();

    let images_raw = B::from_bytes::<f32>(
        images_bytes,
        &[n_samples, 1, 8, 8],
        KindleDType::F32,
        &device,
    )
    .expect("from_bytes images");
    let labels_raw = B::make_labels(label_values);

    let images =
        Tensor::<Dyn, B>::from_raw(images_raw, vec![n_samples, 1, 8, 8]).expect("images tensor");
    let labels: Tensor<Dyn, B, u32, _, NoGrad> =
        Tensor::from_raw(labels_raw, vec![n_samples]).expect("labels tensor");

    // conv1: 1 -> 4 channels, 3x3/stride1/pad1 keeps spatial size at 8x8.
    // pool: 2x2/stride2 halves spatial size to 4x4 -> flattened_dim = 4*4*4 = 64.
    let mut model = SimpleCnn::<B>::new(1, 4, 2, 64).expect("model init");

    // `--bench-telemetry`'s loss-curve-identity check (ARCH-02/TELEM-07)
    // requires the *same* initial weights across the compared runs --
    // `SimpleCnn::new`'s `randn`/`rand`-based weight init is otherwise
    // unseeded and differs across independent calls, which would make any
    // loss-curve difference ambiguous (telemetry effect vs. random-init
    // effect). When `init_state` is provided, load it over the freshly
    // initialized model before training so both compared runs start from
    // bit-identical weights; the non-bench call sites in `main()` pass
    // `None` and are unaffected.
    if let Some(state) = init_state {
        model
            .load_state_dict("", state)
            .expect("load_state_dict (shared init weights)");
    }

    let mut optim = SGD::<B>::new(model.parameters(), lr);
    let ce_loss = CrossEntropyLoss::<Mean>::new();

    let mut losses = Vec::with_capacity(n_epochs);
    let start = std::time::Instant::now();
    for step in 0..n_epochs {
        let output = model.forward(images.clone()).expect("forward");
        let loss = ce_loss
            .forward(&output, &labels)
            .expect("cross_entropy_loss");
        let loss_val: f32 = loss.to_scalar().expect("loss scalar");
        losses.push(loss_val as f64);

        if let Some(r) = reporter {
            r.log_scalar(kindle_telemetry::events::ScalarEvent {
                schema_version: kindle_telemetry::events::CURRENT_SCHEMA_VERSION,
                step,
                name: String::from("loss"),
                value: loss_val as f64,
            });
        }

        let grads = loss.backward().expect("backward");

        // Gradient-norm sampling is strictly opt-in and synchronous: only
        // ever computed when the caller explicitly requests it via
        // `grad_sample_every_n`, and only on the requested cadence -- never
        // an unconditional per-step device-to-host copy (per this plan's
        // must_haves and ARCH-02).
        if let (Some(r), Some(n)) = (reporter, grad_sample_every_n)
            && n > 0
            && step % n == 0
        {
            for (name, var) in model.parameters().iter() {
                let t = B::var_as_tensor::<f32>(var).expect("var_as_tensor");
                if let Some(grad) = B::get_grad::<f32>(&t, &grads.0).expect("get_grad") {
                    // L2 norm of the raw gradient storage, computed directly
                    // via the existing NumericOps/ReductionOps/TensorOps
                    // backend methods (no new backend trait surface):
                    // sqrt(sum(grad * grad)).
                    let grad_sq = B::mul::<f32>(&grad, &grad).expect("mul (grad^2)");
                    let sum_sq = B::sum_all::<f32>(&grad_sq).expect("sum_all (grad^2)");
                    let sum_sq_val = B::float_to_scalar::<f32>(&sum_sq).expect("float_to_scalar");
                    let l2_norm = sum_sq_val.sqrt();

                    r.log_gradient_norm(kindle_telemetry::events::GradientNormEvent {
                        schema_version: kindle_telemetry::events::CURRENT_SCHEMA_VERSION,
                        step,
                        param_name: name.clone(),
                        l2_norm: l2_norm as f32,
                    });
                }
            }
        }

        optim.step(&grads).expect("optimizer step");
    }
    let elapsed = start.elapsed();

    (losses, elapsed, model)
}

/// Runs `train::<NB>()` three ways back-to-back on the identical dataset and
/// hyperparameters -- telemetry off, telemetry on at recommended defaults
/// (file transport only, gradient sampling off), and telemetry on with
/// worst-case gradient sampling every step -- printing wall-clock timing for
/// each, and asserting the off/on-default loss curves are bit-for-bit
/// identical (per this plan's ARCH-02/TELEM-07 proof requirement).
///
/// Returns `true` iff both the throughput-regression bar (run 2 vs run 1,
/// <= 2.0%) and the loss-curve-identity check (run 1 == run 2, exactly) pass.
/// Run 3 (worst-case gradient sampling) is printed but never gates
/// pass/fail -- per RESEARCH.md's Code Example 5, worst-case sampling is
/// explicitly expected to cost more.
fn run_bench_telemetry(
    images_bytes: &[u8],
    labels: &[u32],
    n_samples: usize,
    n_epochs: usize,
    lr: f64,
) -> bool {
    println!("Running --bench-telemetry (TELEM-07/ARCH-02 dogfooding proof)");
    println!();

    // `SimpleCnn::new`'s weight init (`randn`/`rand`) is unseeded, so two
    // independent `train()` calls would start from different initial weights
    // and produce different loss curves regardless of telemetry -- which
    // would make the loss-curve-identity check meaningless (telemetry effect
    // vs. random-init effect, indistinguishable). To isolate telemetry as the
    // only variable, build one throwaway model here purely to snapshot its
    // freshly initialized weights via `state_dict`, then pass that same
    // snapshot as `init_state` into all three runs below so they all start
    // from bit-identical weights.
    let init_model = SimpleCnn::<NB>::new(1, 4, 2, 64).expect("init snapshot model");
    let mut init_state = BTreeMap::new();
    init_model.state_dict("", &mut init_state);

    // Run 1: baseline, telemetry off entirely.
    let (losses_1, elapsed_1, _model_1) = train::<NB>(
        images_bytes,
        labels,
        n_samples,
        n_epochs,
        lr,
        None,
        None,
        Some(&init_state),
    );

    // Run 2: telemetry on at recommended defaults -- file transport only,
    // gradient sampling off. Constructed via the same run_dir/generate_run_id
    // + FileTransport::open + Emitter::new path a real caller would use.
    let run_dir = kindle_telemetry::run_dir::default_run_dir()
        .expect("default_run_dir should succeed for --bench-telemetry");
    let run_id_2 = kindle_telemetry::run_dir::generate_run_id();
    let file_transport_2 = kindle_telemetry::transport::file::FileTransport::open(
        &run_dir.join(format!("{run_id_2}.jsonl")),
    )
    .expect("FileTransport::open should succeed for --bench-telemetry run 2");
    let emitter_2 = kindle_telemetry::emitter::Emitter::new(vec![Box::new(file_transport_2)]);
    let (losses_2, elapsed_2, _model_2) = train::<NB>(
        images_bytes,
        labels,
        n_samples,
        n_epochs,
        lr,
        Some(&emitter_2),
        None,
        Some(&init_state),
    );

    // Run 3: worst case -- telemetry on, gradient-norm sampling every step.
    // A separate Emitter instance, own run_id, same file-transport setup.
    let run_id_3 = kindle_telemetry::run_dir::generate_run_id();
    let file_transport_3 = kindle_telemetry::transport::file::FileTransport::open(
        &run_dir.join(format!("{run_id_3}.jsonl")),
    )
    .expect("FileTransport::open should succeed for --bench-telemetry run 3");
    let emitter_3 = kindle_telemetry::emitter::Emitter::new(vec![Box::new(file_transport_3)]);
    let (_losses_3, elapsed_3, _model_3) = train::<NB>(
        images_bytes,
        labels,
        n_samples,
        n_epochs,
        lr,
        Some(&emitter_3),
        Some(1),
        Some(&init_state),
    );

    let pct_regression =
        (elapsed_2.as_secs_f64() - elapsed_1.as_secs_f64()) / elapsed_1.as_secs_f64() * 100.0;

    println!("telemetry off                         : {elapsed_1:?}");
    println!(
        "telemetry on (default, no grad sample) : {elapsed_2:?}  ({pct_regression:+.2}% vs off)"
    );
    println!("telemetry on (worst case, every step)  : {elapsed_3:?}  (not gated, informational)");
    println!();

    let throughput_ok = pct_regression <= 2.0;
    println!(
        "Throughput regression (<= 2.0% bar, off vs default-on): {}",
        if throughput_ok { "PASS" } else { "FAIL" }
    );

    let loss_curve_identical = losses_1 == losses_2;
    println!(
        "Loss-curve identity (off vs default-on, exact equality): {}",
        if loss_curve_identical { "PASS" } else { "FAIL" }
    );

    throughput_ok && loss_curve_identical
}

/// Runs a single, long-lived, telemetry-emitting training run suitable for a
/// human to attach `kindle-viz` to *while it is still training* (D-03's
/// live-attach proof). Unlike `--bench-telemetry` (three back-to-back runs
/// that finish and exit), this prints the run-id/attach-command up front,
/// then trains for `live_epochs` (200) epochs with a per-epoch sleep so the
/// whole run takes on the order of tens of seconds -- long enough to launch
/// `kindle-viz` mid-run and observe live updates.
///
/// Loss-curve-only proof (no gradient-norm sampling) per D-01/D-02 --
/// gradient-norm panels are Phase 9 scope.
///
/// `train()` has no built-in per-epoch sleep hook and this plan's
/// `files_modified` is scoped to this file only, so the delay is achieved by
/// calling `train::<NB>()` once per single epoch in a loop here, threading
/// the model's `state_dict()` forward between iterations via
/// `StateDict::state_dict`/`load_state_dict` so training genuinely continues
/// (loss keeps decreasing) across the per-epoch sleep rather than restarting
/// from scratch each iteration.
fn run_live(images_bytes: &[u8], labels: &[u32], n_samples: usize, lr: f64) -> anyhow::Result<()> {
    let run_dir = kindle_telemetry::run_dir::default_run_dir()?;
    let run_id = kindle_telemetry::run_dir::generate_run_id();
    let file_transport = kindle_telemetry::transport::file::FileTransport::open(
        &run_dir.join(format!("{run_id}.jsonl")),
    )?;
    let emitter = kindle_telemetry::emitter::Emitter::new(vec![Box::new(file_transport)]);

    println!("live run-id: {run_id}");
    println!("kindle-viz --run-id {run_id}");

    /// Auto-generated documentation for LIVE_EPOCHS.
    const LIVE_EPOCHS: usize = 200;
    let mut state: Option<BTreeMap<String, Tensor<Dyn, NB>>> = None;

    for _ in 0..LIVE_EPOCHS {
        let (_losses, _elapsed, model) = train::<NB>(
            images_bytes,
            labels,
            n_samples,
            1,
            lr,
            Some(&emitter),
            None,
            state.as_ref(),
        );

        let mut next_state = BTreeMap::new();
        model.state_dict("", &mut next_state);
        state = Some(next_state);

        std::thread::sleep(std::time::Duration::from_millis(200));
    }

    emitter.shutdown();

    Ok(())
}

fn main() -> anyhow::Result<()> {
    let live = std::env::args().any(|a| a == "--live");
    let bench_telemetry = std::env::args().any(|a| a == "--bench-telemetry");

    println!("Starting native_training_demo (NATBACK-11 definition-of-done example)");

    let (images, labels, n_samples) = make_dataset();
    let images_bytes = as_f32_bytes(&images);

    let n_epochs = 20;
    let lr = 0.05;

    println!(
        "Training SimpleCnn (conv2d->batch_norm->relu->max_pool2d->flatten->linear) \
         on {n_samples} synthetic 8x8 samples for {n_epochs} epochs (lr={lr})"
    );

    if live {
        return run_live(&images_bytes, &labels, n_samples, lr);
    }

    if bench_telemetry {
        let ok = run_bench_telemetry(&images_bytes, &labels, n_samples, n_epochs, lr);
        if !ok {
            std::process::exit(1);
        }
        return Ok(());
    }

    let (native_losses, native_elapsed, _native_model) = train::<NB>(
        &images_bytes,
        &labels,
        n_samples,
        n_epochs,
        lr,
        None,
        None,
        None,
    );
    let (candle_losses, candle_elapsed, _candle_model) = train::<CB>(
        &images_bytes,
        &labels,
        n_samples,
        n_epochs,
        lr,
        None,
        None,
        None,
    );

    println!();
    for i in 0..n_epochs {
        let nl = native_losses[i];
        let cl = candle_losses[i];
        println!("epoch {i}: native_loss={nl:.6}  candle_loss={cl:.6}");
    }
    println!("NativeBackend: {native_elapsed:?} total | CandleBackend: {candle_elapsed:?} total");

    let native_ok = native_losses.last().unwrap() < native_losses.first().unwrap();
    let candle_ok = candle_losses.last().unwrap() < candle_losses.first().unwrap();
    println!();
    println!(
        "NativeBackend loss decreased first->last: {}",
        if native_ok { "PASS" } else { "FAIL" }
    );
    println!(
        "CandleBackend loss decreased first->last: {}",
        if candle_ok { "PASS" } else { "FAIL" }
    );

    Ok(())
}
