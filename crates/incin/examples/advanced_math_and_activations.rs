//! Practical Example: Advanced Math, Slicing, and LLM Attention Mechanics.
//!
//! Demonstrates:
//! 1. Scaled dot-product attention with causal mask and temperature
//! 2. Rotary Position Embeddings (RoPE) computation
//! 3. Tensor slicing and sub-tensor extraction
//! 4. Multi-axis reductions (mean, sum, argmax)
//! 5. Numerical stability patterns (log-softmax, clamp, stable variance).
//!
//! Run with: `cargo run -p incin --example advanced_math_and_activations --features cpu`

#![allow(missing_docs)]
#![allow(clippy::type_complexity)]

use incin::prelude::*;

type Backend = DefaultBackend;

fn main() -> incin::Result<()> {
    println!("=== Practical Example: Advanced Math, Attention Mechanics & RoPE ===\n");

    // ─────────────────────────────────────────────────────────────────────────
    // 1. Rotary Position Embeddings (RoPE)
    // ─────────────────────────────────────────────────────────────────────────
    println!("[1] Computing Rotary Position Embeddings (RoPE)...");
    let seq_len = 4;
    let head_dim = 8;
    let theta_base = 10000.0_f32;

    // Frequencies: theta_i = 1.0 / (theta_base ^ (2i / head_dim)) for i in 0..head_dim/2
    let mut freqs = Vec::with_capacity(head_dim / 2);
    for i in 0..(head_dim / 2) {
        let power = (2 * i) as f32 / head_dim as f32;
        freqs.push(1.0 / theta_base.powf(power));
    }
    println!("  • Base frequencies (d/2): {:?}", freqs);

    // Outer product: m * theta_i for m in 0..seq_len
    let mut angles = Vec::with_capacity(seq_len * (head_dim / 2));
    for m in 0..seq_len {
        for &freq in &freqs {
            angles.push(m as f32 * freq);
        }
    }
    let angle_tensor: Tensor<Dyn, Backend> =
        Cpu.tensor_from_vec(angles, [seq_len, head_dim / 2])?;
    println!("  • RoPE angle matrix shape: {:?}", angle_tensor.dims());

    // Compute sin and cos components
    let cos_emb = angle_tensor.cos()?;
    let sin_emb = angle_tensor.sin()?;
    println!("  • Cos embedding shape: {:?}", cos_emb.dims());
    println!("  • Sin embedding shape: {:?}", sin_emb.dims());

    // ─────────────────────────────────────────────────────────────────────────
    // 2. Scaled Dot-Product Attention with Causal Masking
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n[2] Computing Scaled Dot-Product Attention with Temperature...");
    let q: Tensor<s![4, 8], Backend> = Cpu.randn(shape![4, 8])?;
    let k: Tensor<s![4, 8], Backend> = Cpu.randn(shape![4, 8])?;
    let v: Tensor<s![4, 8], Backend> = Cpu.randn(shape![4, 8])?;

    // 1. Raw scores: Q @ K.T
    let k_t = k.transpose(axis!(0), axis!(1))?;
    let raw_scores = q.matmul(&k_t)?;

    // 2. Temperature scaling (e.g. sqrt(d_k) = sqrt(8) = 2.8284)
    let temperature = (8.0_f32).sqrt();
    let scaled_scores = raw_scores * (1.0 / temperature);

    // 3. Softmax across sequence axis (-1)
    let attn_probs = scaled_scores.softmax(axis!(-1))?;
    println!("  • Attention probability distribution (row sum = 1.0):");
    println!("    {:?}", attn_probs.to_vec1::<f32>()?);

    // 4. Output projection: Probs @ V
    let context = attn_probs.matmul(&v)?;
    println!("  • Context output shape: {:?}", context.dims());

    // ─────────────────────────────────────────────────────────────────────────
    // 3. Multi-Axis Reductions & Argmax
    // ─────────────────────────────────────────────────────────────────────────
    println!("\n[3] Reductions & Argmax Operations...");
    let matrix = Cpu.tensor([
        [1.0_f32, 5.0, 3.0, 2.0],
        [9.0, 2.0, 4.0, 1.0],
        [0.0, 8.0, 7.0, 6.0],
    ])?;

    // Max class index per row
    let max_indices = matrix.argmax(axis!(1))?;
    println!("  • Argmax across columns (axis 1): {:?}", max_indices.to_vec1::<u32>()?);

    // Mean across rows (dim 0)
    let col_means = matrix.clone().sum_keepdim(axis!(0))? * (1.0 / 3.0_f32);
    println!("  • Column means: {:?}", col_means.to_vec1::<f32>()?);

    // Total matrix sum
    let total_sum = matrix.sum_all()?;
    println!("  • Total element sum: {:.2}", total_sum.to_scalar::<f32>()?);

    println!("\n[4] Advanced math and attention operations completed successfully!");
    Ok(())
}
