use kindle::prelude::*;

fn main() -> anyhow::Result<()> {
    println!("Starting RNN Sequence Prediction Example");
    
    #[cfg(feature = "candle")]
    type Backend = kindle_backends::candle::CandleBackend<f32, Cpu>;
    #[cfg(not(feature = "candle"))]
    type Backend = kindle_core::tensor::backend::DummyBackend<f32, Cpu>;
    
    // Define an RNN sequence model with static dimension shapes
    // Input features: 10, Output features: 20
    let wi = kindle::nn::Linear::<s![10, 20], Backend>::new()?;
    let wh = kindle::nn::Linear::<s![20, 20], Backend>::new()?;
    let cell = kindle::nn::rnn::RNNCell::new(wi, wh);
    let model = kindle::nn::rnn::RNN::<typenum::U10, typenum::U20, Backend>::new(cell);
    
    println!("Created RNN model with In=10, Out=20");
    
    // Batch size = 2, Sequence Length = 5, Features = 10
    // So the input shape is (2, 5, 10)
    // We instantiate it as a statically typed tensor
    let input = Tensor::<s![2, 5, 10], Backend>::zeros(())?;
    println!("Created input sequence tensor with shape: {:?}", input.dims());
    
    println!("Running forward pass over the sequence...");
    let h0 = Tensor::<s![2, 20], Backend>::zeros(())?;
    let output = model.forward((input, h0)).unwrap();
    
    // Output shape should be (2, 5, 20)
    println!("Output sequence shape: {:?}", output.0.dims());
    
    // Save the model weights
    println!("Saving RNN model to Safetensors format...");
    model.save(Format::Safetensors, std::path::Path::new("rnn_model.safetensors")).unwrap();
    println!("Saved successfully!");
    
    Ok(())
}
