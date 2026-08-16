# Data loading

`Dataset` is a random-access source (`len` + `get(index) -> Result<Option<Item>, DataError>`),
and `Collate` turns a `Vec<Item>` into whatever batched form your model
actually wants, such as a tensor, a tuple of tensors, or another batch type.

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

`DataLoader::new` takes `(dataset, collate_fn, batch_size)`, with the collate
function before the batch size. Iterate with `&loader` (it implements
`IntoIterator` by
reference, so the loader itself is reusable across epochs).

For the common case where a batch should remain a vector of samples, use the
default builder:

```rust,ignore
let loader = DataLoader::builder(Toy)
    .batch_size(4)
    .workers(2)
    .shuffle(true)
    .build()?;
```

Custom collation remains available through `DataLoader::new` and
`DataLoader::builder_with_collate`.

The default collator batches scalar samples as `Vec<T>`, batches tuple samples
field-wise, and stacks compatible tensor samples along a leading batch axis.
For example, `(Tensor<A>, u8)` becomes `(Tensor<Dyn>, Vec<u8>)`, while
`(Tensor<A>, Tensor<B>)` becomes `(Tensor<Dyn>, Tensor<Dyn>)`. Vector-valued
samples remain `Vec<Vec<T>>`. Shape or backend failures while stacking tensors
are returned as `DataError::InvalidBatch`.

Errors are values, not end-of-epoch signals. A dataset or worker failure is
returned as `Err(DataError)` from iteration and must be handled by the caller;
it is never silently converted to `None`. With `num_workers == 0`, constructing
the iterator performs no dataset reads, and each `next()` fetches and collates
only its next batch synchronously. Worker-backed iteration keeps its explicit
cancellation and error propagation semantics.

## Model-ready MNIST batches

The MNIST dataset keeps storage and download logic independent from a backend,
while its provided collator performs target-aware conversion at the data
loader boundary. It returns normalized image tensors with shape
`[batch, 1, 28, 28]` and integer `u8` label tensors with `NoGrad`.

```rust,no_run
use incin::prelude::*;
use incin_data::vision::mnist::{MnistCollate, MnistDataset};
use incin_data::DataLoader;

type Backend = incin_backends::cpu::CpuBackendImpl;

let dataset = MnistDataset::new("./data/mnist", true)?;
let loader = DataLoader::builder_with_collate(dataset, MnistCollate::<Backend>::new())
    .batch_size(32)
    .shuffle(true)
    .build()?;

for batch in &loader {
    let (images, labels) = batch.map_err(|error| incin::Error::Msg(error.to_string()))?;
    let _logits = images;
    let _classes = labels;
}
# Ok::<(), incin::Error>(())
```

This path avoids flattening samples into host vectors and rebuilding tensors in
the training loop. The complete example is in
`crates/incin/examples/mnist_training.rs`.

## Transforms

`incin_data::transforms` has the common image-augmentation set:
`CenterCrop`, `Compose`, `Normalize`, `RandomHorizontalFlip`, and `Scale`,
implementing a shared `Transform` trait, composable with `Compose`.
