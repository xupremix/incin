use kindle::prelude::*;
use typenum::consts::*;

#[module]
pub struct BasicBlock<B: Backend> {
    pub conv1: kindle::nn::Conv2d<(usize, usize, U3, U1, U1, U1), B>,
    pub bn1: kindle::nn::BatchNorm2d<(usize,), B>,
    pub conv2: kindle::nn::Conv2d<(usize, usize, U3, U1, U1, U1), B>,
    pub bn2: kindle::nn::BatchNorm2d<(usize,), B>,
}

impl<B: Backend> BasicBlock<B>
where
    B::FloatElem: ConstDType,
    B::Device: ConstDevice,
{
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        _stride: usize,
        _device: &KindleDevice,
    ) -> Result<Self> {
        Ok(Self {
            conv1: kindle::nn::Conv2d::<(usize, usize, U3, U1, U1, U1), B>::new_with((
                out_channels,
                in_channels,
            ))?,
            bn1: kindle::nn::BatchNorm2d::<(usize,), B>::new_with((out_channels,), 1e-5, 0.1)?,
            conv2: kindle::nn::Conv2d::<(usize, usize, U3, U1, U1, U1), B>::new_with((
                out_channels,
                out_channels,
            ))?,
            bn2: kindle::nn::BatchNorm2d::<(usize,), B>::new_with((out_channels,), 1e-5, 0.1)?,
        })
    }
}

impl<B: Backend> BasicBlock<B> {
    pub fn forward(&self, x: Tensor<Dyn, B>) -> Result<Tensor<Dyn, B>> {
        let out = self.conv1.forward(x.clone())?;
        let out = self.bn1.forward(out)?;
        let out = out.relu()?;
        let out = self.conv2.forward(out)?;
        let out = self.bn2.forward(out)?;

        let out = out.add(&x)?;
        Ok(out.relu()?)
    }
}

#[module]
pub struct ResNet<B: Backend> {
    pub conv1: kindle::nn::Conv2d<(usize, usize, U7, U2, U3, U1), B>,
    pub bn1: kindle::nn::BatchNorm2d<(usize,), B>,
    pub layer1: BasicBlock<B>,
    pub fc: kindle::nn::Linear<Dyn, B>,
}

impl<B: Backend> ResNet<B>
where
    B::FloatElem: ConstDType,
    B::Device: ConstDevice,
{
    pub fn new(num_classes: usize, device: &KindleDevice) -> Result<Self> {
        Ok(Self {
            conv1: kindle::nn::Conv2d::<(usize, usize, U7, U2, U3, U1), B>::new_with((64, 3))?,
            bn1: kindle::nn::BatchNorm2d::<(usize,), B>::new_with((64,), 1e-5, 0.1)?,
            layer1: BasicBlock::<B>::new(64, 64, 1, device)?,
            fc: kindle::nn::Linear::<Dyn, B>::new_with((64, num_classes))?,
        })
    }
}

impl<B: Backend> ResNet<B> {
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
