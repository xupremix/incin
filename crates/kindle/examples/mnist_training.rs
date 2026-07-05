use kindle::nn::StateDict;
use kindle::prelude::*;
use kindle_data::vision::mnist::MnistDataset;
use kindle_data::Dataset;
use std::path::PathBuf;

#[module]
pub struct MLP<B: Backend> {
    fc1: Linear<Dyn, B>,
    fc2: Linear<Dyn, B>,
}

impl<B: Backend> MLP<B>
where
    B::DType: ConstDType,
    B::Device: ConstDevice,
{
    pub fn new() -> Result<Self> {
        let fc1 = Linear::<Dyn, B>::new(784, 128)?;
        let fc2 = Linear::<Dyn, B>::new(128, 10)?;
        Ok(Self { fc1, fc2 })
    }
}

#[forward]
impl<B: Backend> MLP<B> {
    fn forward(&self, x: &Tensor<Dyn, B>) -> Result<Tensor<Dyn, B>> {
        let x = self.fc1.forward(x.clone())?.relu()?;
        let x = self.fc2.forward(x)?;
        Ok(x)
    }
}

fn main() -> anyhow::Result<()> {
    println!("Starting MNIST Training Example");
    
    #[cfg(feature = "candle")]
    type Backend = kindle_backends::candle::CandleBackend<f32, Cpu>;
    #[cfg(not(feature = "candle"))]
    type Backend = kindle_core::tensor::backend::DummyBackend<f32, Cpu>;
    
    let device = KindleDevice::cpu();
    
    // 1. Dataset loading
    let data_dir = PathBuf::from("./data/mnist");
    println!("Loading dataset into {:?}...", data_dir);
    let train_data = MnistDataset::new(&data_dir, true)?;
    println!("Loaded {} training images", train_data.len());
    
    // 2. Model definition
    let mut model = MLP::<Backend>::new().unwrap();
    
    // 3. Fake Training Loop (just a forward pass on first batch for demo)
    if let Some((img, label)) = train_data.get(0) {
        // img is 784 f32 elements
        let input = Tensor::<Dyn, Backend>::zeros(vec![1, 784]).unwrap(); // Dummy input representing the image
        // We could load the real data: Backend::from_bytes(...) but zeros is fine for the example skeleton
        
        println!("Forward pass running...");
        let output = model.forward(&input).unwrap();
        println!("Output shape: {:?}", output.dims());
    }
    
    // 4. Save Checkpoint
    println!("Saving checkpoint...");
    let mut serializer = kindle_core::serialize::SafetensorsSerializer::new(std::path::Path::new("mnist_model.safetensors"));
    model.save_to(&mut serializer).unwrap();
    println!("Saved successfully!");
    
    Ok(())
}
