# Metrics

Metrics accumulate over batches rather than compute a single value from
tensors directly - call `update` per batch, read `value()` whenever you want
the running score, `reset()` between epochs.

```rust,no_run
use incin::prelude::*;

fn main() -> Result<()> {
    let logits = tensor![[2.0_f32, 1.0], [0.5, 3.0]]?;
    let preds_idx = logits.argmax(axis!(1))?.to_vec1::<u32>()?; // argmax's index dtype defaults to u32
    let preds: Vec<usize> = preds_idx.iter().map(|&v| v as usize).collect();
    let labels: Vec<usize> = vec![0, 1];

    let mut acc = Accuracy::new();
    acc.update(&preds, &labels);
    println!("accuracy: {}", acc.value());

    acc.reset();
    Ok(())
}
```

`update` takes plain `&[usize]` class-index slices, not tensors - pull the
values off the device first (`to_vec1`), same as any other host-side
readout. `Precision`, `Recall`, `F1Score`, `MSE`, and `ConfusionMatrix`
follow the same `Metric` trait shape (`reset`, `value`).
