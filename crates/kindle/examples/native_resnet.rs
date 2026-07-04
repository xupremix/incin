use kindle::prelude::*;
use kindle_backends::prelude::*;

#[module]
pub struct BasicBlock<B: Backend<Dyn>> {
    pub conv1: kindle::nn::Conv2d<Dyn, U1, U1, B>,
    pub bn1: kindle::nn::BatchNorm2d<Dyn, B>,
    pub conv2: kindle::nn::Conv2d<Dyn, U1, U1, B>,
    pub bn2: kindle::nn::BatchNorm2d<Dyn, B>,
}

impl<B: Backend<Dyn>> BasicBlock<B> {
    pub fn new(in_channels: usize, out_channels: usize, stride: usize, device: &KindleDevice) -> Result<Self> {
        Ok(Self {
            conv1: kindle::nn::Conv2d::<Dyn, U1, U1, B>::new(in_channels, out_channels, 3, 3)?, // Wait! new signature is (cout, cin, kh, kw). So (out_channels, in_channels, 3, 3)
            bn1: kindle::nn::BatchNorm2d::<Dyn, B>::new(out_channels, device)?,
            conv2: kindle::nn::Conv2d::<Dyn, U1, U1, B>::new(out_channels, out_channels, 3, 3)?,
            bn2: kindle::nn::BatchNorm2d::<Dyn, B>::new(out_channels, device)?,
        })
    }
}

#[forward]
impl<B: Backend<Dyn>> BasicBlock<B> {
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
pub struct ResNet<B: Backend<Dyn>> {
    pub conv1: kindle::nn::Conv2d<Dyn, U1, U1, B>,
    pub bn1: kindle::nn::BatchNorm2d<Dyn, B>,
    pub layer1: BasicBlock<B>,
    pub fc: kindle::nn::Linear<Dyn, B>,
}

impl<B: Backend<Dyn>> ResNet<B> {
    pub fn new(num_classes: usize, device: &KindleDevice) -> Result<Self> {
        Ok(Self {
            conv1: kindle::nn::Conv2d::<Dyn, U1, U1, B>::new(64, 3, 7, 7)?, // cout=64, cin=3
            bn1: kindle::nn::BatchNorm2d::<Dyn, B>::new(64, device)?,
            layer1: BasicBlock::new(64, 64, 1, device)?,
            fc: kindle::nn::Linear::<Dyn, B>::new(64, num_classes)?,
        })
    }
}

#[forward]
impl<B: Backend<Dyn>> ResNet<B> {
    pub fn forward(&self, x: Tensor<Dyn, B>) -> Result<Tensor<Dyn, B>> {
        let x = self.conv1.forward(x)?;
        let x = self.bn1.forward(x)?;
        let x = x.relu()?;
        
        let x = self.layer1.forward(x)?;
        
        let x = x.mean_dim::<2>()?.mean_dim::<2>()?; // Global average pool over H, W

        
        let out = self.fc.forward(x)?;
        Ok(out)
    }
}

fn main() -> Result<()> {
    let device = KindleDevice::cpu();
    let model = ResNet::<CandleBackend>::new(1000, &device)?;
    
    // Dummy input: [Batch, Channels, Height, Width]
    let input = Tensor::<Dyn, CandleBackend>::zeros(vec![1, 3, 224, 224])?;
    
    let out = model.forward(input)?;
    println!("ResNet output shape: {:?}", out.dims());
    assert_eq!(out.dims(), &[1, 1000]);
    
    Ok(())
}
