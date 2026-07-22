use kindle::Backend as _;
use kindle::Flatten;
use kindle::prelude::*;
use kindle_data::vision::mnist::MnistDataset;
use kindle_data::{Collate, DataLoader, Dataset};
use std::path::PathBuf;

type Backend = kindle_backends::cpu::CpuBackendImpl;

/// Mnist collate.
struct MnistCollate;

impl Collate<(Vec<f32>, u8)> for MnistCollate {
    /// Output.
    type Output = (Tensor<Dyn, Backend>, Tensor<Dyn, Backend>);

    /// Collate.
    fn collate(&self, batch: Vec<(Vec<f32>, u8)>) -> Self::Output {
        let batch_size = batch.len();
        let mut images = Vec::with_capacity(batch_size * 784);
        let mut labels = Vec::with_capacity(batch_size);

        for (img, label) in batch {
            images.extend_from_slice(&img);
            labels.push(label as f32); // F32 target tensor for CrossEntropyLoss
        }

        let images_bytes = unsafe {
            core::slice::from_raw_parts(
                images.as_ptr() as *const u8,
                images.len() * core::mem::size_of::<f32>(),
            )
        };

        let labels_bytes = unsafe {
            core::slice::from_raw_parts(
                labels.as_ptr() as *const u8,
                labels.len() * core::mem::size_of::<f32>(),
            )
        };

        let device = DeviceId::cpu();
        let images_raw = Backend::from_bytes::<f32>(
            images_bytes,
            &[batch_size, 1, 28, 28],
            DTypeId::F32,
            &device,
        )
        .unwrap();
        let labels_raw =
            Backend::from_bytes::<f32>(labels_bytes, &[batch_size], DTypeId::F32, &device).unwrap();

        (
            Tensor::<Dyn, Backend>::from_raw(images_raw, vec![batch_size, 1, 28, 28]).unwrap(),
            Tensor::<Dyn, Backend>::from_raw(labels_raw, vec![batch_size]).unwrap(),
        )
    }
}

fn main() -> kindle::Result<()> {
    println!("Starting MNIST Training Example");

    // 1. Dataset loading
    let data_dir = PathBuf::from("./data/mnist");
    println!("Loading dataset into {:?}...", data_dir);
    let train_data = MnistDataset::new(&data_dir, true)?;
    println!("Loaded {} training images", train_data.len());

    // Create DataLoader
    let dataloader = DataLoader::new(train_data, MnistCollate, 32)
        .with_shuffle(true)
        .with_num_workers(0); // Using 0 for simple blocking execution

    // 2. Model definition (MLP using the seq! macro and Flatten)
    let model = seq![
        Flatten::<1, 3>::new(), // Flattens (B, 1, 28, 28) -> (B, 784)
        Linear::<Dyn, Backend>::build((784, 128))?,
        ReLU,
        Linear::<Dyn, Backend>::build((128, 10))?
    ];

    // 3. Optimizer setup
    let mut optim = kindle::AdamW::<Backend>::new(model.parameters(), 0.001);

    // 4. Real Training Loop
    println!("Starting training...");
    let mut batch_idx = 0;
    for (images, labels) in &dataloader {
        // Forward pass
        let output = model.forward(images)?;

        // Compute loss
        let loss = output.cross_entropy_loss(&labels)?;

        // Backward pass
        let grads = loss.backward()?;

        // Optimizer step
        optim.step(&grads)?;

        if batch_idx % 2 == 0 {
            println!("Processed batch {}, loss = {:.4}", batch_idx, loss.to_scalar::<f32>()?);
        }
        batch_idx += 1;

        if batch_idx >= 10 {
            break; // Stop early for demo
        }
    }

    // 5. Save Checkpoint
    println!("Saving checkpoint to Safetensors format...");
    model
        .save(
            Format::Safetensors,
            std::path::Path::new("mnist_model.safetensors"),
        )
        .unwrap();
    println!("Saved successfully!");

    Ok(())
}
