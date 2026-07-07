use kindle::prelude::*;
use typenum::{U0, U1, U3};

type Features = Sequential<Conv2d<(usize, usize, U3, U1, U0, U1)>, ReLU>;

#[module]
pub struct SimpleCNN {
    features: Features,
    classifier: Linear<Dyn>,
}

impl SimpleCNN {
    pub fn new() -> Result<Self> {
        // Init 16 out_channels, 1 in_channel
        let conv = Conv2d::<(usize, usize, U3, U1, U0, U1)>::new_with((16, 1), ())?;
        let relu = ReLU;
        let features = Sequential(conv, relu);

        let classifier = Linear::<Dyn>::new_with((16 * 26 * 26, 10), ())?;

        Ok(Self {
            features,
            classifier,
        })
    }

    pub fn forward(&self, x: Tensor<Dyn>) -> Result<Tensor<Dyn>> {
        // 1. Feature extraction (Conv2d -> ReLU)
        let f = self.features.forward(x)?;

        // 2. Flatten (B, 16, 26, 26) -> (B, 10816)
        let flat = f.flatten::<1, 3>()?;

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
