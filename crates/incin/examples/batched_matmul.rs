//! Example: exercises the documented API around `main`.
#![allow(clippy::type_complexity)]

use incin::prelude::*;

fn main() -> Result<()> {
    // 1. Batched Matrix multiplied by a Batched Matrix (Static)
    let t1 = Cpu.randn(shape![3, 10, 20])?;
    let t2 = Cpu.randn(shape![20, 5])?;

    // The output shape is statically verified as [3, 10, 5]
    let t3 = t1.matmul(&t2)?;
    println!("t1 (static) shape: {:?}", t1.dims());
    println!("t2 (static) shape: {:?}", t2.dims());
    println!("t3 (static) shape after matmul: {:?}", t3.dims());

    // 2. Dynamic batch dimension broadcasting
    // [Batch, 10, 20] where Batch=32 at runtime
    let batch = 32;
    let dyn_t1 = Cpu.randn(shape![batch, 10, 20])?;
    // [20, 5] (static weight matrix)
    let static_weights = Cpu.randn(shape![20, 5])?;

    // The output shape retains runtime batch and static features: [32, 10, 5]
    let dyn_t3 = dyn_t1.matmul(&static_weights)?;

    println!("\ndyn_t1 shape: {:?}", dyn_t1.dims());
    println!("static_weights shape: {:?}", static_weights.dims());
    println!("dyn_t3 shape after broadcasted matmul: {:?}", dyn_t3.dims());

    // 3. Multi-Head Attention Style (4 dimensions: [Batch, Heads, SeqLen, HeadDim])
    let q = Cpu.randn(shape![2, 4, 16, 8])?;
    let k_t = Cpu.randn(shape![2, 4, 8, 16])?;

    let attn_scores = q.matmul(&k_t)?;
    println!("\nq shape: {:?}", q.dims());
    println!("k_t shape: {:?}", k_t.dims());
    println!("attn_scores shape: {:?}", attn_scores.dims());

    Ok(())
}
