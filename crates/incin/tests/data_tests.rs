use incin::prelude::*;
use incin_data::{DataError, DataLoader, Dataset};

#[test]
/// Test dataset.
fn test_dataset() {
    // The public data types are reachable through their documented paths.
}

#[cfg(feature = "cpu")]
struct TensorPairs;

#[cfg(feature = "cpu")]
impl Dataset for TensorPairs {
    type Item = (Tensor<s![2], DefaultBackend>, u8);

    fn len(&self) -> usize {
        4
    }

    fn get(&self, index: usize) -> std::result::Result<Option<Self::Item>, DataError> {
        if index >= self.len() {
            return Ok(None);
        }
        let tensor = match Tensor::<s![2], DefaultBackend>::full(index as f32, ()) {
            Ok(tensor) => tensor,
            Err(error) => return Err(DataError::Dataset(error.to_string())),
        };
        Ok(Some((tensor, index as u8)))
    }
}

#[cfg(feature = "cpu")]
#[test]
fn default_collate_stacks_tensor_tuple_fields() {
    let loader = DataLoader::builder(TensorPairs)
        .batch_size(2)
        .build()
        .unwrap();
    let mut batches = (&loader).into_iter();
    let (images, labels) = batches.next().unwrap().unwrap();
    assert_eq!(images.dims().as_ref(), &[2, 2]);
    assert_eq!(labels, vec![0, 1]);
}
