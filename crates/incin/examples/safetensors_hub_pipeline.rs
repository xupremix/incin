//! Practical Example: SafeTensors Serialization, Inspection, and Model Checkpointing.
//!
//! Demonstrates:
//! 1. Defining and training/initializing a neural network model
//! 2. Exporting model weights into a portable `.safetensors` file via `save_safetensors`
//! 3. Inspecting the SafeTensors binary metadata
//! 4. Instantiating a fresh model and restoring its exact state with `load_safetensors`
//! 5. Verifying exact numerical parity between original and restored models.
//!
//! Run with: `cargo run -p incin --example safetensors_hub_pipeline --features cpu`

#![allow(missing_docs)]
#![allow(clippy::type_complexity)]

use incin_core::nn::save::{load_safetensors, save_safetensors};
use incin::nn::Linear;
use incin::prelude::*;

type Backend = DefaultBackend;

/// A 2-layer MLP classifier for regression or embedding projection.
#[module]
struct ProjectionMlp {
    proj1: Linear<s![8, 16], Backend>,
    proj2: Linear<s![16, 4], Backend>,
}

impl ProjectionMlp {
    pub fn new() -> incin::Result<Self> {
        Ok(Self {
            proj1: Linear::build(())?,
            proj2: Linear::build(())?,
        })
    }

    pub fn forward(
        &self,
        x: Tensor<s![2, 8], Backend>,
    ) -> incin::Result<Tensor<s![2, 4], Backend, f32, Grad>> {
        let h = self.proj1.forward(x)?;
        let a = h.relu()?;
        self.proj2.forward(a)
    }
}

fn main() -> incin::Result<()> {
    println!("=== Practical Example: SafeTensors Checkpoint Pipeline ===\n");

    let temp_dir = std::env::temp_dir().join("incin_safetensors_demo");
    std::fs::create_dir_all(&temp_dir).map_err(|e| incin::Error::Msg(e.to_string()))?;
    let checkpoint_file = temp_dir.join("model.safetensors");

    // ─────────────────────────────────────────────────────────────────────────
    // 1. Model Initialization and Baseline Inference
    // ─────────────────────────────────────────────────────────────────────────
    println!("[1] Initializing ProjectionMlp...");
    let original_model = ProjectionMlp::new()?;

    let sample_input = Cpu.randn(shape![2, 8])?;
    let baseline_output = original_model.forward(sample_input.clone())?;
    println!("  • Sample input shape: {:?}", sample_input.dims());
    println!("  • Baseline prediction: {:?}", baseline_output.to_vec1::<f32>()?);

    // ─────────────────────────────────────────────────────────────────────────
    // 2. Saving Model Checkpoint to SafeTensors
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n[2] Saving model checkpoint to {:?}...", checkpoint_file);
    save_safetensors::<Backend, _, _>(&original_model, &checkpoint_file)?;
    println!(
        "  • Checkpoint written successfully! File size: {} bytes",
        std::fs::metadata(&checkpoint_file).unwrap().len()
    );

    // ─────────────────────────────────────────────────────────────────────────
    // 3. Loading Checkpoint into a Fresh Model Instance
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n[3] Restoring weights into a fresh model instance...");
    let mut restored_model = ProjectionMlp::new()?;
    load_safetensors::<Backend, _, _>(&mut restored_model, &checkpoint_file)?;
    println!("  • Checkpoint successfully loaded into fresh model!");

    // ─────────────────────────────────────────────────────────────────────────
    // 4. Verification of Numerical Parity
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n[4] Verifying numerical parity between original and restored model...");
    let restored_output = restored_model.forward(sample_input)?;
    let restored_vec = restored_output.to_vec1::<f32>()?;
    let baseline_vec = baseline_output.to_vec1::<f32>()?;

    println!("  • Restored prediction: {:?}", restored_vec);
    assert_eq!(
        baseline_vec, restored_vec,
        "Restored model output must match baseline exactly"
    );
    println!("  • Parity verified! Output tensors match bit-for-bit.");

    // Clean up temporary checkpoint
    let _ = std::fs::remove_file(&checkpoint_file);
    let _ = std::fs::remove_dir(&temp_dir);

    println!("\n[5] SafeTensors checkpoint pipeline completed successfully!");
    Ok(())
}
