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
For example, `(Tensor<A>, u8)` becomes `(Vec<Tensor<A>>, Vec<u8>)` unless the
tensor field is itself collated by a custom tuple collator. Vector-valued
samples remain `Vec<Vec<T>>`. Shape or backend failures while stacking tensors
are returned as `DataError::InvalidBatch`.

Errors are values, not end-of-epoch signals. A dataset or worker failure is
returned as `Err(DataError)` from iteration and must be handled by the caller;
it is never silently converted to `None`. With `num_workers == 0`, constructing
the iterator performs no dataset reads, and each `next()` fetches and collates
only its next batch synchronously. Worker-backed iteration keeps its explicit
cancellation and error propagation semantics.

## A model-specific collate function

The default MNIST path receives `(Vec<Vec<f32>>, Vec<u8>)`, so the example
does not require custom loader plumbing. Applications that want tensors
created inside the loader can provide a model-specific `Collate` implementation
like this:

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

`incin_data::transforms` has the common image-augmentation set  -
`CenterCrop`, `Compose`, `Normalize`, `RandomHorizontalFlip`, `Scale`  -
implementing a shared `Transform` trait, composable with `Compose`.
