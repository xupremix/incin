use kindle::prelude::*;
use kindle_data::vision::mnist::MnistDataset;
use kindle_data::Dataset;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    println!("Starting MNIST Training Example");
    
    #[cfg(feature = "candle")]
    type Backend = kindle_backends::candle::CandleBackend<f32, Cpu>;
    #[cfg(not(feature = "candle"))]
    type Backend = kindle_core::tensor::backend::DummyBackend<f32, Cpu>;
    
    let _device = KindleDevice::cpu();
    
    // 1. Dataset loading
    let data_dir = PathBuf::from("./data/mnist");
    println!("Loading dataset into {:?}...", data_dir);
    let train_data = MnistDataset::new(&data_dir, true)?;
    println!("Loaded {} training images", train_data.len());
    
    // 2. Model definition (MLP using the seq! macro and default initialization)
    let model = seq![
        Linear::<Dyn, Backend>::new(784, 128)?,
        ReLU,
        Linear::<Dyn, Backend>::new(128, 10)?
    ];
    
    // 3. Optimizer setup
    // Fetch all parameters from the model and pass them to AdamW
    let mut _optim = kindle::optim::AdamW::<Backend>::new(model.parameters(), 0.001);
    
    // 4. Fake Training Loop (just a forward pass on first batch for demo)
    if let Some((_img, _label)) = train_data.get(0) {
        // Dummy input representing a batch of 1 image (flattened 28x28)
        let input = Tensor::<Dyn, Backend>::zeros(vec![1, 784]).unwrap(); 
        
        println!("Running forward pass...");
        let output = model.forward(input).unwrap();
        println!("Output shape: {:?}", output.dims());
    }
    
    // 5. Save Checkpoint (using Unified API)
    println!("Saving checkpoint to Safetensors format...");
    model.save(Format::Safetensors, std::path::Path::new("mnist_model.safetensors")).unwrap();
    println!("Saved successfully!");
    
    Ok(())
}
