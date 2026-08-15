use incin::prelude::*;
use incin_data::vision::mnist::MnistDataset;
use incin_data::{BatchResult, Collate, DataError, DataLoader, Dataset};
use std::path::PathBuf;

type Backend = incin_backends::cpu::CpuBackendImpl;

/// Mnist collate.
struct MnistCollate;

impl Collate<(Vec<f32>, u8)> for MnistCollate {
    /// Output.
    type Output = (Tensor<Dyn, Backend>, Tensor<Dyn, Backend, f32, Grad>);

    /// Collate.
    fn collate(&self, batch: Vec<(Vec<f32>, u8)>) -> BatchResult<Self::Output> {
        let batch_size = batch.len();
        let mut images = Vec::with_capacity(batch_size * 784);
        let mut labels = Vec::with_capacity(batch_size);

        for (img, label) in batch {
            images.extend_from_slice(&img);
            labels.push(label as f32); // F32 target tensor for CrossEntropyLoss
        }

        Ok((
            Tensor::<Dyn, Backend>::from_slice(&images, vec![batch_size, 1, 28, 28])
                .map_err(|error| DataError::Dataset(error.to_string()))?,
            Tensor::<Dyn, Backend>::from_slice(&labels, vec![batch_size])
                .map_err(|error| DataError::Dataset(error.to_string()))?
                .require_grad(),
        ))
    }
}

fn main() -> incin::Result<()> {
    println!("Starting MNIST Training Example");

    // 1. Dataset loading
    let data_dir = PathBuf::from("./data/mnist");
    println!("Loading dataset into {:?}...", data_dir);
    let train_data = MnistDataset::new(&data_dir, true)?;
    println!("Loaded {} training images", train_data.len());

    // Create DataLoader
    let dataloader = DataLoader::new(train_data, MnistCollate, 32)?
        .with_shuffle(true)
        .with_num_workers(0); // Using 0 for simple blocking execution

    // 2. Model definition (MLP using the seq! macro and Flatten)
    let model = seq![
        Flatten::<Next<Here>, Next<Next<Here>>>::new(), // Flattens (B, 1, 28, 28) -> (B, 784)
        Linear::<Dyn, Backend>::build((784, 128))?,
        ReLU,
        Linear::<Dyn, Backend>::build((128, 10))?
    ];

    // 3. Optimizer setup
    let mut optim = AdamW::<Backend>::from_module(&model, 0.001)?;

    // 4. Real Training Loop
    println!("Starting training...");
    let mut batch_idx = 0;
    for batch in &dataloader {
        let (images, labels) = batch.map_err(|error| incin::Error::Msg(error.to_string()))?;
        // Forward pass
        let output = model.forward(images)?;

        // Compute loss
        let loss = output.cross_entropy_loss(&labels)?;

        // Backward pass
        let grads = loss.backward()?;

        // Optimizer step
        optim.step(&grads)?;

        if batch_idx % 2 == 0 {
            println!(
                "Processed batch {}, loss = {:.4}",
                batch_idx,
                loss.to_scalar::<f32>()?
            );
        }
        batch_idx += 1;

        if batch_idx >= 10 {
            break; // Stop early for demo
        }
    }

    // Checkpoint save/load is covered by the stateful model fixture and Book;
    // this deliberately data-focused example stops after the training proof.

    Ok(())
}
