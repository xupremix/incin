use kindle::prelude::*;
use kindle_data::prelude::*;
use rayon::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    println!("--- Parallel DataLoader & HuggingFace Example ---");
    
    // 1. Dataloader Extension
    let items: Vec<i32> = (0..10_000).collect();
    let iter = items.into_iter();
    
    let start = Instant::now();
    // `.into_par_loader()` effortlessly utilizes Rayon's threadpool to process data
    let sum: i32 = iter.into_par_loader()
        .map(|x: i32| {
            // Simulate complex parsing/loading
            let mut temp = x;
            for _ in 0..100 {
                temp = temp.wrapping_add(1);
            }
            temp
        })
        .sum();
        
    println!("Processed 10,000 items in parallel in {:?}", start.elapsed());
    println!("Sum: {}", sum);
    
    println!("\n(Note: HuggingFace hub loading works asynchronously via `HuggingFaceHub::load_safetensors`, see `crates/kindle-data/src/hf.rs` for the async API!)");
    
    Ok(())
}
