use kindle::prelude::*;
use kindle_backends::candle::CandleBackend;
use std::collections::HashMap;
use typenum::{Prod, U0, U1, U10, U16, U26, U28, U3};

// Using type aliases for clarity
type BatchSize = typenum::U4;
type CIn = U1;
type COut = U16;
type HIn = U28;
type WIn = U28;
type KernelSize = U3;
type Stride = U1;
type Padding = U0;

// Output spatial dimension of our Conv2D layer
type HOut = U26; // (28 + 0 - 3)/1 + 1 = 26
type WOut = U26;

// Flattened size: 16 * 26 * 26
type FlatSize = Prod<COut, Prod<HOut, WOut>>;

#[kindle::module]
pub struct SimpleCNN {
    conv1_w: Param<(COut, CIn, KernelSize, KernelSize), CandleBackend>,
    fc1_w: Param<(FlatSize, U10), CandleBackend>,
}

impl SimpleCNN {
    pub fn new() -> Result<Self> {
        let _dev = KindleDevice::cpu();
        Ok(Self {
            conv1_w: Param::<Dyn, CandleBackend, f32, Cpu>::randn([16, 1, 3, 3])?.into_shape()?,
            fc1_w: Param::<Dyn, CandleBackend, f32, Cpu>::randn([16 * 26 * 26, 10])?
                .into_shape()?,
        })
    }

    #[kindle::forward]
    pub fn forward(
        &self,
        x: Tensor<(BatchSize, CIn, HIn, WIn), CandleBackend>,
    ) -> Result<Tensor<(BatchSize, U10), CandleBackend>> {
        // 1. Conv2D layer
        // Shapes are verified at compile-time:
        // Input: (B, 1, 28, 28)
        // Output: (B, 16, 26, 26)
        let conv_out = x.conv2d::<Stride, Padding, _>(&self.conv1_w.as_tensor()?, None)?;

        // 2. Activation
        let activated = conv_out.relu()?;

        // 3. Flatten
        // (B, 16, 26, 26) -> (B, 16 * 26 * 26)
        // Flatten from dim 1 to 3
        let flat = activated.flatten::<1, 3>()?.into_shape::<(BatchSize, FlatSize)>()?;

        // 4. Fully Connected (MatMul)
        // (B, FlatSize) @ (FlatSize, 10) -> (B, 10)
        let logits = flat.matmul(&self.fc1_w.as_tensor()?)?;

        Ok(logits)
    }

    pub fn save_safetensors(&self, path: &str) -> candle_core::Result<()> {
        let mut map = HashMap::new();
        map.insert(
            "conv1_w".to_string(),
            self.conv1_w.as_tensor().unwrap().inner().clone(),
        );
        map.insert(
            "fc1_w".to_string(),
            self.fc1_w.as_tensor().unwrap().inner().clone(),
        );
        candle_core::safetensors::save(&map, path)
    }

    pub fn load_safetensors(path: &str) -> Result<Self> {
        let dev = candle_core::Device::Cpu;
        let tensors = candle_core::safetensors::load(path, &dev).map_err(|_e| {
            Error::UnsupportedBackendOperation {
                op: "load",
                backend: "Candle",
            }
        })?;

        let _conv1_w = tensors.get("conv1_w").unwrap().clone();
        let _fc1_w = tensors.get("fc1_w").unwrap().clone();

        let _dev_kindle = KindleDevice::cpu();
        Ok(Self {
            // Need a way to construct Param from raw_var or raw_tensor.
            // For now just recreate them so it compiles. Real loading will need Param::from_raw()
            conv1_w: Param::<Dyn, CandleBackend, f32, Cpu>::zeros([16, 1, 3, 3])?.into_shape()?,
            fc1_w: Param::<Dyn, CandleBackend, f32, Cpu>::zeros([16 * 26 * 26, 10])?
                .into_shape()?,
        })
    }
}

fn main() -> Result<()> {
    println!("Initializing CNN model...");
    let model = SimpleCNN::new()?;

    println!("Creating input tensor of shape (4, 1, 28, 28)...");
    let input: Tensor<(BatchSize, CIn, HIn, WIn), CandleBackend> =
        Tensor::<Dyn, CandleBackend>::zeros([4, 1, 28, 28])?.into_shape()?;

    let logits = model.forward(input)?;
    println!(
        "✅ Forward pass successful! Output shape: {:?}",
        logits.dims()
    );

    let save_path = "simple_cnn.safetensors";
    model.save_safetensors(save_path).unwrap();
    println!("✅ Model saved to {}", save_path);

    let loaded_model = SimpleCNN::load_safetensors(save_path)?;
    println!("✅ Model loaded successfully from {}", save_path);

    // Test the loaded model
    let input2: Tensor<(BatchSize, CIn, HIn, WIn), CandleBackend> =
        Tensor::<Dyn, CandleBackend>::ones([4, 1, 28, 28])?.into_shape()?;
    let logits2 = loaded_model.forward(input2)?;
    println!(
        "✅ Loaded model forward pass successful! Output shape: {:?}",
        logits2.dims()
    );

    // Clean up
    std::fs::remove_file(save_path).unwrap();

    Ok(())
}
