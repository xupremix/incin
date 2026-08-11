//! Backend training demo: trains a small CNN classifier
//! (conv2d -> batch_norm -> relu -> max_pool2d -> flatten -> linear) end-to-end
//! (forward -> cross_entropy_loss -> backward -> optimizer step) for multiple
//! epochs on `CpuBackendImpl<Cpu>`.

#[macro_use]
extern crate alloc;

use incin::backend_authoring::SupportsDType;
use incin::optim::OptimizerBackend;
use incin::prelude::*;
use incin::prelude::{CrossEntropyLoss, Mean};
use incin_backends::cpu::CpuBackendImpl;
use incin_core::exec::catalog::op;
use incin_core::nn::param::ParameterInit;
use incin_core::tensor::backend::Execute;

/// The CPU backend type alias.
type NB = CpuBackendImpl;

// ── Model ────────────────────────────────────────────────────────────────────

/// A small CNN classifier: conv2d -> batch_norm -> relu -> max_pool2d -> flatten -> linear.
#[module]
pub struct SimpleCnn<B: Backend> {
    pub conv1: incin_core::prelude::Conv2d<s![dyn, dyn, 3, 1, 1, 1], B, incin_core::prelude::False>,
    pub bn1: incin::BatchNorm2d<s![dyn], B>,
    #[module(ignore)]
    pub pool: incin::MaxPool2d<typenum::U2, typenum::U2>,
    pub fc: incin::Linear<Dyn, B>,
}

impl<B: Backend + incin_core::nn::param::ParameterInit<f32>> SimpleCnn<B>
where
    B: SupportsDType<f32> + SupportsDType<u32>,
    B::Device: ConstDevice,
{
    pub fn new(
        in_channels: usize,
        conv_out_channels: usize,
        num_classes: usize,
        flattened_dim: usize,
    ) -> Result<Self> {
        Ok(Self {
            conv1: incin_core::prelude::Conv2d::<
                s![dyn, dyn, 3, 1, 1, 1],
                B,
                incin_core::prelude::False,
            >::build((conv_out_channels, in_channels))?,
            bn1: incin::BatchNorm2d::<s![dyn], B>::build((conv_out_channels, 1e-5, 0.1))?,
            pool: incin::MaxPool2d::<typenum::U2, typenum::U2>::new()?,
            fc: incin::Linear::<Dyn, B>::build((flattened_dim, num_classes))?,
        })
    }
}

impl<
    B: Backend
        + Execute<op::TransposeExact>
        + Execute<op::ReshapeExact>
        + incin_core::tensor::backend::Execute<op::MatMulExact>
        + Execute<op::Add>
        + Execute<op::Relu>
        + Execute<op::Conv2dExact>
        + Execute<op::MaxPool2d>
        + Execute<op::BatchNorm>
        + Execute<op::CrossEntropyLoss>,
> SimpleCnn<B>
where
    <B as Execute<op::Add>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::Relu>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::Conv2dExact>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::MatMulExact>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::TransposeExact>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::ReshapeExact>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::MaxPool2d>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::BatchNorm>>::Output: Into<B::Storage<f32>>,
{
    pub fn forward(&self, x: Tensor<Dyn, B>) -> Result<Tensor<Dyn, B, f32, Grad>> {
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

fn make_dataset() -> (Vec<f32>, Vec<u32>, usize) {
    let mut rng: u64 = 0x1234_5678_9abc_def0;
    let mut next_f32 = || -> f32 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((rng >> 33) as f32) / ((1u64 << 31) as f32) - 0.5
    };

    let n_samples = 16;
    let in_channels = 1;
    let (h, w) = (8, 8);
    let n_elems = n_samples * in_channels * h * w;

    let mut images = Vec::with_capacity(n_elems);
    let mut labels = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let label = (i % 2) as u32;
        labels.push(label);
        let bias = if label == 1 { 0.5 } else { -0.5 };
        for _ in 0..(in_channels * h * w) {
            images.push(next_f32() * 0.1 + bias);
        }
    }

    (images, labels, n_samples)
}

// ── Training loop ────────────────────────────────────────────────────────────

fn train<B>(
    images: &[f32],
    labels: &[u32],
    n_samples: usize,
    n_epochs: usize,
    lr: f64,
) -> (Vec<f32>, std::time::Duration)
where
    B: Backend
        + SupportsDType<f32>
        + SupportsDType<u32>
        + ParameterInit<f32>
        + OptimizerBackend<f32>
        + Execute<op::TensorFromData>
        + Execute<op::MatMulExact>
        + Execute<op::Add>
        + Execute<op::Relu>
        + Execute<op::Conv2dExact>
        + Execute<op::TransposeExact>
        + Execute<op::ReshapeExact>
        + Execute<op::MaxPool2d>
        + Execute<op::BatchNorm>
        + Execute<op::CrossEntropyLoss>,
    B::Device: ConstDevice,
    <B as Execute<op::Add>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::Relu>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::Conv2dExact>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::MatMulExact>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::TransposeExact>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::ReshapeExact>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::MaxPool2d>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::BatchNorm>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::CrossEntropyLoss>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::TensorFromData>>::Output: Into<B::Storage<f32>> + Into<B::Storage<u32>>,
{
    let in_channels = 1;
    let conv_out_channels = 4;
    let num_classes = 2;
    let (h, w) = (8, 8);
    let pool_h = h / 2;
    let pool_w = w / 2;
    let flattened_dim = conv_out_channels * pool_h * pool_w;

    let model =
        SimpleCnn::<B>::new(in_channels, conv_out_channels, num_classes, flattened_dim).unwrap();
    let mut sgd = SGD::<B>::new(model.parameters(), lr);

    let start = std::time::Instant::now();
    let mut losses = Vec::with_capacity(n_epochs);

    let target_shape = vec![n_samples, in_channels, h, w];

    let labels_u32 = labels.to_vec();

    for _epoch in 0..n_epochs {
        let x = Tensor::<Dyn, B>::from_slice(images, target_shape.clone()).unwrap();
        let targets = Tensor::<Dyn, B, u32>::from_slice(&labels_u32, vec![n_samples]).unwrap();

        let logits = model.forward(x).unwrap();

        let loss_fn = CrossEntropyLoss::<Mean>::new();
        let loss = loss_fn.forward(&logits, &targets).unwrap();
        let loss_val: f32 = loss.to_vec1().unwrap()[0];
        losses.push(loss_val);

        let grads = loss.backward().unwrap();
        sgd.step(&grads).unwrap();
    }

    let elapsed = start.elapsed();
    (losses, elapsed)
}

fn main() -> anyhow::Result<()> {
    let (images, labels, n_samples) = make_dataset();
    let n_epochs = 20;
    let lr = 0.05;

    println!(
        "Training SimpleCnn (conv2d->batch_norm->relu->max_pool2d->flatten->linear) \
         on {n_samples} synthetic 8x8 samples for {n_epochs} epochs (lr={lr})"
    );

    let (native_losses, native_elapsed) = train::<NB>(&images, &labels, n_samples, n_epochs, lr);

    println!();
    for (i, nl) in native_losses.iter().enumerate().take(n_epochs) {
        println!("epoch {i}: loss={nl:.6}");
    }
    println!("CpuBackendImpl: {native_elapsed:?} total");

    let native_ok = native_losses.last().unwrap() < native_losses.first().unwrap();
    println!();
    println!(
        "CpuBackendImpl loss decreased first->last: {}",
        if native_ok { "PASS" } else { "FAIL" }
    );

    Ok(())
}
