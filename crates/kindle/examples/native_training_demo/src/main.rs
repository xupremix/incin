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

use kindle::nn::{CrossEntropyLoss, Mean};
use kindle::optim::SGD;
use kindle::prelude::*;
use kindle_backends::candle::CandleBackend;
use kindle_native::NativeBackend;

type NB = NativeBackend<f32, Cpu>;
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
    pub conv1: kindle::nn::Conv2d<(usize, usize, U3, U1, U1, U1), B, kindle::nn::optional::False>,
    pub bn1: kindle::nn::BatchNorm2d<(usize,), B>,
    // MaxPool2d is a zero-sized, stateless module (no parameters, no
    // device-bound buffers) and does not implement `ToDevice` -- it is
    // excluded from the `#[module]`-derived Parameters/StateDict/ToDevice
    // impls, which is semantically correct since it has nothing to collect
    // or move across devices.
    #[module(ignore)]
    pub pool: kindle::nn::MaxPool2d<U2, U2>,
    pub fc: kindle::nn::Linear<Dyn, B>,
}

impl<B: Backend> SimpleCnn<B>
where
    B::FloatElem: ConstDType,
    B::Device: ConstDevice,
{
    pub fn new(
        in_channels: usize,
        conv_out_channels: usize,
        num_classes: usize,
        flattened_dim: usize,
    ) -> Result<Self> {
        Ok(Self {
            conv1: kindle::nn::Conv2d::<
                (usize, usize, U3, U1, U1, U1),
                B,
                kindle::nn::optional::False,
            >::new_with((conv_out_channels, in_channels))?,
            bn1: kindle::nn::BatchNorm2d::<(usize,), B>::new_with((conv_out_channels,), 1e-5, 0.1)?,
            pool: kindle::nn::MaxPool2d::<U2, U2>::new()?,
            fc: kindle::nn::Linear::<Dyn, B>::new_with((flattened_dim, num_classes))?,
        })
    }
}

impl<B: Backend> SimpleCnn<B> {
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
    const N: usize = 32;
    const HW: usize = 8;
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
    fn make_labels(values: &[u32]) -> Self::Storage<u32>;
}

impl MakeLabels for NB {
    fn make_labels(values: &[u32]) -> Self::Storage<u32> {
        kindle_native::storage::NativeStorage::from_contiguous(
            kindle_native::storage::NativeBuffer::U32(values.to_vec()),
            vec![values.len()],
        )
    }
}

impl MakeLabels for CB {
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
fn train<B: MakeLabels>(
    images_bytes: &[u8],
    label_values: &[u32],
    n_samples: usize,
    n_epochs: usize,
    lr: f64,
) -> (Vec<f64>, std::time::Duration)
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
    let model = SimpleCnn::<B>::new(1, 4, 2, 64).expect("model init");
    let mut optim = SGD::<B>::new(model.parameters(), lr);
    let ce_loss = CrossEntropyLoss::<Mean>::new();

    let mut losses = Vec::with_capacity(n_epochs);
    let start = std::time::Instant::now();
    for _ in 0..n_epochs {
        let output = model.forward(images.clone()).expect("forward");
        let loss = ce_loss
            .forward(&output, &labels)
            .expect("cross_entropy_loss");
        let loss_val: f32 = loss.to_scalar().expect("loss scalar");
        losses.push(loss_val as f64);

        let grads = loss.backward().expect("backward");
        optim.step(&grads).expect("optimizer step");
    }
    let elapsed = start.elapsed();

    (losses, elapsed)
}

fn main() -> anyhow::Result<()> {
    println!("Starting native_training_demo (NATBACK-11 definition-of-done example)");

    let (images, labels, n_samples) = make_dataset();
    let images_bytes = as_f32_bytes(&images);

    let n_epochs = 20;
    let lr = 0.05;

    println!(
        "Training SimpleCnn (conv2d->batch_norm->relu->max_pool2d->flatten->linear) \
         on {n_samples} synthetic 8x8 samples for {n_epochs} epochs (lr={lr})"
    );

    let (native_losses, native_elapsed) =
        train::<NB>(&images_bytes, &labels, n_samples, n_epochs, lr);
    let (candle_losses, candle_elapsed) =
        train::<CB>(&images_bytes, &labels, n_samples, n_epochs, lr);

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
