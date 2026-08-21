#![allow(clippy::type_complexity)]

extern crate alloc;

use incin::prelude::*;

/// The compute backend to execute operations on.
type Backend = DefaultBackend;

// ── Model Definition ─────────────────────────────────────────────────────────

/// A simple multi-layer perceptron (MLP) demonstrating static shape safety.
#[module]
struct Mlp {
    l1: Linear<s![10, 20], Backend>,
    l2: Linear<s![20, 20], Backend>,
    l3: Linear<s![20, 10], Backend>,
}

impl Mlp {
    /// Initialize layers with default weights and biases.
    pub fn new() -> Result<Self> {
        Ok(Self {
            l1: Linear::build(())?,
            l2: Linear::build(())?,
            l3: Linear::build(())?,
        })
    }

    /// Forward pass through the network: Linear -> ReLU -> Linear -> ReLU -> Linear.
    pub fn forward(
        &self,
        x: Tensor<s![4, 10], Backend>,
    ) -> Result<Tensor<s![4, 10], Backend, f32, Grad>> {
        // 1. First hidden layer + ReLU
        let h1 = self.l1.forward(x)?;
        let a1 = h1.relu()?;

        // 2. Second hidden layer + ReLU
        let h2 = self.l2.forward(a1)?;
        let a2 = h2.relu()?;

        // 3. Output layer
        let out = self.l3.forward(a2)?;
        Ok(out)
    }
}

// ── Main Entrypoint ──────────────────────────────────────────────────────────

fn main() -> Result<()> {
    println!("=== Incin Target API & Tensor Creation Showcase ===\n");

    // 1. Target-First Allocation (PyTorch / NumPy equivalents)
    // --------------------------------------------------------
    // Standard normal distribution ~ N(0, 1)  (PyTorch: torch.randn(4, 10))
    let normal_input = Cpu.randn(shape![4, 10])?;
    println!(
        "• Cpu.randn(shape![4, 10]): shape {:?}",
        normal_input.dims()
    );

    // Uniform random in [0, 1), one value per feature  (PyTorch: torch.rand(10))
    let bias = Cpu.rand(shape![10])?;
    println!("• Cpu.rand(shape![10]):  shape {:?}", bias.dims());

    // Constant fill  (PyTorch: torch.full((2, 3), 7.0))
    let full = Cpu.full(shape![2, 3], 7.0)?;
    println!("• Cpu.full(shape![2, 3], 7.0): shape {:?}", full.dims());

    // Step sequences  (PyTorch: torch.arange(0, 8, 2))
    let stepped = Cpu.arange(shape![4], 0.0, 2.0)?;
    println!(
        "• Cpu.arange(shape![4], 0.0, 2.0): shape {:?}",
        stepped.dims()
    );

    // Linearly spaced values  (PyTorch: torch.linspace(0, 1, 5))
    let spaced = Cpu.linspace(shape![5], 0.0, 1.0)?;
    println!(
        "• Cpu.linspace(shape![5], 0.0, 1.0): shape {:?}",
        spaced.dims()
    );

    // From literal arrays with static shapes  (PyTorch: torch.tensor([[1., 2.], [3., 4.]]))
    let literal_matrix = Cpu.tensor([[1.0_f32, 2.0], [3.0, 4.0]])?;
    println!(
        "• Cpu.tensor([[1., 2.], [3., 4.]]): shape {:?}",
        literal_matrix.dims()
    );

    // 2. Arithmetic & Broadcasting
    // ----------------------------
    // scaled_input is [4, 10]; bias is [10]. The trailing dimensions match,
    // so bias broadcasts across the batch dimension: every row gets the
    // same per-feature offset added to it.
    let scaled_input = &normal_input * 2.0;
    let offset_input = &scaled_input + &bias;
    println!(
        "\n• Broadcasting: [4, 10] * 2.0 + [10] -> shape {:?}",
        offset_input.dims()
    );

    // 3. Neural Network Execution
    // ---------------------------
    println!("\nInitializing MLP model...");
    let model = Mlp::new()?;

    println!("Running forward pass with batch size 4...");
    let logits = model.forward(offset_input)?;

    println!("✅ Forward pass successful!");
    println!("Output shape: {:?}", logits.dims());
    println!("Output sample:\n{logits}");

    Ok(())
}
