//! Example: exercises the documented API around `main`.
#![cfg(feature = "cpu")]

// Keep the executable proof and the integration proof on one canonical source.
#[path = "../tests/transformer_block.rs"]
mod transformer_proof;

fn main() -> incin::Result<()> {
    transformer_proof::cpu_transformer_forward_backward_adamw_and_state_roundtrip()?;
    println!("Transformer forward, backward, AdamW, and state roundtrip passed");
    Ok(())
}
