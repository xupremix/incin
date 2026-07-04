use kindle::prelude::*;
use kindle_backends::candle::CandleBackend;
use kindle_core::nn::{Conv2d, Linear, ReLU, Sequential};
use typenum::{U0, U1};

type Features = Sequential<Conv2d<Dyn, U1, U0, CandleBackend>, ReLU>;

#[kindle::module]
pub struct SimpleCNN {
    features: Features,
    classifier: Linear<Dyn, CandleBackend>,
}

impl SimpleCNN {
    pub fn new() -> Result<Self> {
        // Init 16 out_channels, 1 in_channel, 3x3 kernel
        let conv = Conv2d::<Dyn, U1, U0, CandleBackend>::new(16, 1, 3, 3)?;
        let relu = ReLU;
        let features = Sequential(conv, relu);

        let classifier = Linear::<Dyn, CandleBackend>::new(16 * 26 * 26, 10)?;

        Ok(Self {
            features,
            classifier,
        })
    }

    #[kindle::forward]
    pub fn forward(&self, x: Tensor<Dyn, CandleBackend>) -> Result<Tensor<Dyn, CandleBackend>> {
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
    let input: Tensor<Dyn, CandleBackend> = Tensor::<Dyn, CandleBackend>::zeros([4, 1, 28, 28])?;

    let logits = model.forward(input)?;
    println!(
        "✅ Forward pass successful! Output shape: {:?}",
        logits.dims()
    );

    Ok(())
}
