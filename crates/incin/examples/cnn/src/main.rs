extern crate alloc;

use incin::prelude::*;

/// Features.
type Features = Sequential<Conv2d<s![dyn, dyn, 3, 1, 0, 1]>, ReLU>;

#[module(no_shape_info)]
/// Simple cnn.
pub struct SimpleCNN {
    features: Features,
    classifier: Linear<Dyn>,
}

impl SimpleCNN {
    /// New.
    pub fn new() -> Result<Self> {
        // Init 16 out_channels, 1 in_channel
        let conv = Conv2d::<s![dyn, dyn, 3, 1, 0, 1]>::build((16, 1))?;
        let relu = ReLU;
        let features = Sequential(conv, relu);

        let classifier = Linear::<Dyn>::build((16 * 26 * 26, 10))?;

        Ok(Self {
            features,
            classifier,
        })
    }

    /// Forward.
    pub fn forward(&self, x: Tensor<Dyn>) -> Result<Tensor<Dyn, DefaultBackend, f32, Grad>> {
        // 1. Feature extraction (Conv2d -> ReLU)
        let f = self.features.forward(x)?;

        // 2. Flatten (B, 16, 26, 26) -> (B, 10816)
        let flat = f.flatten_runtime(1, 3)?;

        // 3. Classification
        let logits = self.classifier.forward(flat)?;

        Ok(logits)
    }
}

fn main() -> Result<()> {
    println!("Initializing CNN model...");
    let model = SimpleCNN::new()?;

    println!("Creating input tensor of shape (4, 1, 28, 28)...");
    let input: Tensor<Dyn> = Tensor::<Dyn>::zeros([4, 1, 28, 28])?;

    let logits = model.forward(input)?;
    println!(
        "✅ Forward pass successful! Output shape: {:?}",
        logits.dims()
    );

    Ok(())
}
