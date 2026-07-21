use kindle::prelude::*;

#[module]
/// Core abstraction for `BasicBlock` within the Kindle framework.
pub struct BasicBlock<B: Backend> {
    /// Core abstraction for `conv1` within the Kindle framework.
    pub conv1: kindle::Conv2d<s![dyn, dyn, 3, 1, 1, 1], B>,
    /// Core abstraction for `bn1` within the Kindle framework.
    pub bn1: kindle::BatchNorm2d<s![dyn], B>,
    /// Core abstraction for `conv2` within the Kindle framework.
    pub conv2: kindle::Conv2d<s![dyn, dyn, 3, 1, 1, 1], B>,
    /// Core abstraction for `bn2` within the Kindle framework.
    pub bn2: kindle::BatchNorm2d<s![dyn], B>,
}

impl<B: Backend> BasicBlock<B>
where
    B::FloatElem: ConstDType,
    B::Device: ConstDevice,
{
    /// Core abstraction for `new` within the Kindle framework.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        _stride: usize,
        _device: &KindleDevice,
    ) -> Result<Self> {
        Ok(Self {
            conv1: kindle::Conv2d::<s![dyn, dyn, 3, 1, 1, 1], B>::new_with((
                out_channels,
                in_channels,
            ))?,
            bn1: kindle::BatchNorm2d::<s![dyn], B>::new_with((out_channels,), 1e-5, 0.1)?,
            conv2: kindle::Conv2d::<s![dyn, dyn, 3, 1, 1, 1], B>::new_with((
                out_channels,
                out_channels,
            ))?,
            bn2: kindle::BatchNorm2d::<s![dyn], B>::new_with((out_channels,), 1e-5, 0.1)?,
        })
    }
}

impl<B: Backend> BasicBlock<B> {
    /// Core abstraction for `forward` within the Kindle framework.
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

#[module]
/// Core abstraction for `ResNet` within the Kindle framework.
pub struct ResNet<B: Backend> {
    /// Core abstraction for `conv1` within the Kindle framework.
    pub conv1: kindle::Conv2d<s![dyn, dyn, 7, 2, 3, 1], B>,
    /// Core abstraction for `bn1` within the Kindle framework.
    pub bn1: kindle::BatchNorm2d<s![dyn], B>,
    /// Core abstraction for `layer1` within the Kindle framework.
    pub layer1: BasicBlock<B>,
    /// Core abstraction for `fc` within the Kindle framework.
    pub fc: kindle::Linear<Dyn, B>,
}

impl<B: Backend> ResNet<B>
where
    B::FloatElem: ConstDType,
    B::Device: ConstDevice,
{
    /// Core abstraction for `new` within the Kindle framework.
    pub fn new(num_classes: usize, device: &KindleDevice) -> Result<Self> {
        Ok(Self {
            conv1: kindle::Conv2d::<s![dyn, dyn, 7, 2, 3, 1], B>::new_with((64, 3))?,
            bn1: kindle::BatchNorm2d::<s![dyn], B>::new_with((64,), 1e-5, 0.1)?,
            layer1: BasicBlock::<B>::new(64, 64, 1, device)?,
            fc: kindle::Linear::<Dyn, B>::new_with((64, num_classes))?,
        })
    }
}

impl<B: Backend> ResNet<B> {
    /// Core abstraction for `forward` within the Kindle framework.
    pub fn forward(&self, x: Tensor<Dyn, B>) -> Result<Tensor<Dyn, B>> {
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
