use incin::MnistTargetExt;
use incin::prelude::*;
use incin_data::Dataset;
use incin_data::vision::mnist::MnistDataset;
use std::path::PathBuf;

type Backend = incin_backends::cpu::CpuBackendImpl;

fn main() -> incin::Result<()> {
    println!("Starting MNIST Training Example");

    // 1. Dataset loading
    let data_dir = PathBuf::from("./data/mnist");
    println!("Loading dataset into {:?}...", data_dir);
    let train_data = MnistDataset::new(&data_dir, true)?;
    println!("Loaded {} training images", train_data.len());

    // Create DataLoader
    let dataloader = train_data
        .loader_on(Cpu)
        .batch_size(32)
        .shuffle(true)
        .workers(0)
        .build()?;

    // 2. Model definition (MLP using the seq! macro and Flatten)
    let model = seq![
        Flatten::new(1isize, -1isize), // Flattens (B, 1, 28, 28) -> (B, 784)
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
        let batch = batch.map_err(|error| incin::Error::Msg(error.to_string()))?;
        let (images, labels) = batch;
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
