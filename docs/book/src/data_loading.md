# Data loading

`Dataset` is a random-access source (`len` + `get(index) -> Result<Option<Item>, DataError>`),
and `Collate` turns a `Vec<Item>` into whatever batched form your model
actually wants — a tensor, a tuple of tensors, or anything else.

```rust,no_run
use incin::prelude::*;
use incin_data::{Collate, DataLoader, Dataset};

struct Toy;

impl Dataset for Toy {
    type Item = f32;
    fn len(&self) -> usize {
        10
    }
    fn get(&self, index: usize) -> std::result::Result<Option<Self::Item>, incin_data::DataError> {
        Ok(Some(index as f32))
    }
}

struct SumCollate;

impl Collate<f32> for SumCollate {
    type Output = f32;
    fn collate(&self, batch: Vec<f32>) -> incin_data::BatchResult<f32> {
        Ok(batch.iter().sum())
    }
}

fn main() -> Result<()> {
    let loader = DataLoader::new(Toy, SumCollate, 4)?; // dataset, collate_fn, batch_size

    let mut total = 0.0;
    for batch in &loader {
        total += batch.map_err(|error| incin::Error::Msg(error.to_string()))?;
    }
    assert_eq!(total, 45.0); // 0+1+...+9
    Ok(())
}
```

`DataLoader::new` takes `(dataset, collate_fn, batch_size)` — collate before
size, easy to get backwards coming from a framework that orders them the
other way. Iterate with `&loader` (it implements `IntoIterator` by
reference, so the loader itself is reusable across epochs).

## A realistic collate function

The MNIST example in the repository (`crates/incin/examples/mnist_training.rs`)
shows the shape a real `Collate` impl takes: gathering a `Vec<(Vec<f32>,
u8)>` batch into two tensors.

```rust,ignore
struct MnistCollate;

impl Collate<(Vec<f32>, u8)> for MnistCollate {
    type Output = (Tensor<Dyn, Backend>, Tensor<Dyn, Backend>);

    fn collate(&self, batch: Vec<(Vec<f32>, u8)>) -> incin_data::BatchResult<Self::Output> {
        let batch_size = batch.len();
        let mut images = Vec::with_capacity(batch_size * 784);
        let mut labels = Vec::with_capacity(batch_size);

        for (img, label) in batch {
            images.extend_from_slice(&img);
            labels.push(label as f32);
        }

        // ... build tensors from the flattened Vecs (see the full example
        // for the from_bytes plumbing) ...
        # Ok(unimplemented!())
    }
}
```

`Item` is `Dyn`-shaped here on purpose: batch size varies with the last,
short batch of an epoch, so the collated tensor's shape is only known at
run time even though every individual image is a fixed `28x28`.

## Transforms

`incin_data::transforms` has the common image-augmentation set —
`CenterCrop`, `Compose`, `Normalize`, `RandomHorizontalFlip`, `Scale` —
implementing a shared `Transform` trait, composable with `Compose`.
