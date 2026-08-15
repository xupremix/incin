#![allow(clippy::type_complexity)]

use incin::backend_authoring::{Execute, SupportsDType, VariableBackend, operations::op};
use incin::prelude::*;
use incin_core::nn::param::ParameterInit;

#[module(no_shape_info)]
/// Basic block.
pub struct BasicBlock<B: VariableBackend> {
    /// Conv1.
    pub conv1: incin::Conv2d<s![dyn, dyn, 3, 1, 1, 1], B>,
    /// Bn1.
    pub bn1: incin::BatchNorm2d<s![dyn], B>,
    /// Conv2.
    pub conv2: incin::Conv2d<s![dyn, dyn, 3, 1, 1, 1], B>,
    /// Bn2.
    pub bn2: incin::BatchNorm2d<s![dyn], B>,
}

impl<B: VariableBackend + ParameterInit<f32>> BasicBlock<B>
where
    B: SupportsDType<f32>,
    B::Device: ConstDevice,
{
    /// New.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        _stride: usize,
        _device: &DeviceId,
    ) -> Result<Self> {
        Ok(Self {
            conv1: incin::Conv2d::<s![dyn, dyn, 3, 1, 1, 1], B>::build((
                out_channels,
                in_channels,
            ))?,
            bn1: incin::BatchNorm2d::<s![dyn], B>::build((out_channels, 1e-5, 0.1))?,
            conv2: incin::Conv2d::<s![dyn, dyn, 3, 1, 1, 1], B>::build((
                out_channels,
                out_channels,
            ))?,
            bn2: incin::BatchNorm2d::<s![dyn], B>::build((out_channels, 1e-5, 0.1))?,
        })
    }
}

impl<
    B: VariableBackend
        + Execute<op::Add>
        + Execute<op::Relu>
        + Execute<op::Conv2dExact>
        + Execute<op::ReshapeExact>
        + Execute<op::BatchNorm>,
> BasicBlock<B>
where
    <B as Execute<op::Add>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::Relu>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::Conv2dExact>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::ReshapeExact>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::BatchNorm>>::Output: Into<B::Storage<f32>>,
{
    /// Forward.
    pub fn forward(&self, x: Tensor<Dyn, B>) -> Result<Tensor<Dyn, B>> {
        let out = self.conv1.forward(x.clone())?;
        let out = self.bn1.forward(out)?;
        let out = out.relu()?;
        let out = self.conv2.forward(out)?;
        let out = self.bn2.forward(out)?;

        let out = out.add(&x)?;
        out.relu()
    }
}

#[module(no_shape_info)]
/// Res net.
pub struct ResNet<B: VariableBackend> {
    /// Conv1.
    pub conv1: incin::Conv2d<s![dyn, dyn, 7, 2, 3, 1], B>,
    /// Bn1.
    pub bn1: incin::BatchNorm2d<s![dyn], B>,
    /// Layer1.
    pub layer1: BasicBlock<B>,
    /// Fc.
    pub fc: incin::Linear<Dyn, B>,
}

impl<B: VariableBackend + ParameterInit<f32>> ResNet<B>
where
    B: SupportsDType<f32>,
    B::Device: ConstDevice,
{
    /// New.
    pub fn new(num_classes: usize, device: &DeviceId) -> Result<Self> {
        Ok(Self {
            conv1: incin::Conv2d::<s![dyn, dyn, 7, 2, 3, 1], B>::build((64, 3))?,
            bn1: incin::BatchNorm2d::<s![dyn], B>::build((64, 1e-5, 0.1))?,
            layer1: BasicBlock::<B>::new(64, 64, 1, device)?,
            fc: incin::Linear::<Dyn, B>::build((64, num_classes))?,
        })
    }
}

impl<
    B: VariableBackend
        + Execute<op::Add>
        + Execute<op::Relu>
        + Execute<op::Conv2dExact>
        + Execute<op::MatMulExact>
        + Execute<op::TransposeExact>
        + Execute<op::ReshapeExact>
        + Execute<op::BatchNorm>,
> ResNet<B>
where
    <B as Execute<op::MatMulExact>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::Add>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::Relu>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::Conv2dExact>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::TransposeExact>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::ReshapeExact>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::BatchNorm>>::Output: Into<B::Storage<f32>>,
{
    /// Forward.
    pub fn forward(&self, x: Tensor<Dyn, B>) -> Result<Tensor<Dyn, B, f32, Grad>> {
        let x = self.conv1.forward(x)?;
        let x = self.bn1.forward(x)?;
        let x = x.relu()?;

        let x = self.layer1.forward(x)?;

        // global average pool logic skipped for brevity, just flatten
        let dims = x.dims();
        let b = dims[0];
        let c = dims[1];
        let x = x.try_reshape(vec![b, c * dims[2] * dims[3]])?;

        self.fc.forward(x)
    }
}

fn main() -> Result<()> {
    // This is just a compilation test basically
    Ok(())
}
