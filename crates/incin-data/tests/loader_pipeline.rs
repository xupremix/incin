//! End-to-end loader pipeline tests through the public `incin-data` API.
//!
//! These exercise what a downstream consumer links against: a custom
//! [`Dataset`], the default collation policy, the builder, iteration
//! semantics including short batches and error propagation, and the
//! transform composition surface. No backend feature is required.

use incin_data::{BatchResult, Collate, DataError, DataLoader, Dataset, Transform};

/// A deterministic in-memory dataset of squared integers.
struct Squares {
    len: usize,
    fail_from: Option<usize>,
}

impl Squares {
    fn new(len: usize) -> Self {
        Self {
            len,
            fail_from: None,
        }
    }

    fn failing_at(len: usize, fail_from: usize) -> Self {
        Self {
            len,
            fail_from: Some(fail_from),
        }
    }
}

impl Dataset for Squares {
    type Item = i32;

    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, index: usize) -> core::result::Result<Option<Self::Item>, DataError> {
        if index >= self.len {
            return Ok(None);
        }
        if let Some(from) = self.fail_from
            && index >= from
        {
            return Err(DataError::Dataset(format!(
                "sample {index} is corrupt in this fixture"
            )));
        }
        let value = i32::try_from(index)
            .map(|v| v * v)
            .map_err(|_| DataError::Dataset("index does not fit i32".into()))?;
        Ok(Some(value))
    }
}

/// Collects every batch an iterator yields, failing the test on error rows.
fn drain<'a, L, B>(loader: &'a L) -> Vec<B>
where
    &'a L: IntoIterator<Item = BatchResult<B>>,
{
    loader
        .into_iter()
        .map(|batch| batch.expect("fixture dataset must not produce errors"))
        .collect()
}

#[test]
fn batches_cover_every_sample_in_order_without_workers() {
    let loader = DataLoader::from_dataset(Squares::new(10), 4).expect("valid configuration");
    let batches = drain(&loader);

    assert_eq!(batches.len(), 3);
    assert_eq!(batches[0], vec![0, 1, 4, 9]);
    assert_eq!(batches[1], vec![16, 25, 36, 49]);
    // The trailing partial batch is preserved by default; dropping it is an
    // explicit opt-out, not the silent behavior.
    assert_eq!(batches[2], vec![64, 81]);
}

#[test]
fn batch_size_larger_than_the_dataset_yields_one_short_batch() {
    let loader = DataLoader::from_dataset(Squares::new(3), 64).expect("valid configuration");
    let batches = drain(&loader);

    assert_eq!(batches, vec![vec![0, 1, 4]]);
}

#[test]
fn drop_last_discards_only_the_trailing_partial_batch() {
    let loader = DataLoader::builder(Squares::new(10))
        .batch_size(4)
        .drop_last(true)
        .build()
        .expect("valid configuration");
    let batches = drain(&loader);

    assert_eq!(batches.len(), 2);
    assert_eq!(batches[1], vec![16, 25, 36, 49]);
}

#[test]
fn dataset_errors_surface_as_iteration_errors_rather_than_end_of_data() {
    let loader =
        DataLoader::from_dataset(Squares::failing_at(6, 3), 2).expect("valid configuration");

    let mut seen = 0;
    for batch in &loader {
        match batch {
            Ok(values) => {
                seen += values.len();
                assert_eq!(values, [0, 1].to_vec());
            }
            Err(DataError::Dataset(message)) => {
                assert!(message.contains("corrupt"), "unexpected message: {message}");
                return;
            }
            Err(other) => panic!("expected a dataset error, got {other:?}"),
        }
    }
    panic!("the loader ended without reporting the failing sample after {seen} items");
}

#[test]
fn default_collation_batches_tuple_datasets_elementwise() {
    struct Pairs;
    impl Dataset for Pairs {
        type Item = (i32, u8);
        fn len(&self) -> usize {
            5
        }
        fn get(&self, index: usize) -> core::result::Result<Option<Self::Item>, DataError> {
            if index >= 5 {
                return Ok(None);
            }
            Ok(Some((index as i32, (index % 2) as u8)))
        }
    }

    let loader = DataLoader::builder(Pairs)
        .batch_size(3)
        .build()
        .expect("valid configuration");
    let batches = drain(&loader);

    // Default collation is elementwise: one vector per tuple slot.
    assert_eq!(batches[0], (vec![0, 1, 2], vec![0u8, 1, 0]));
    assert_eq!(batches[1], (vec![3, 4], vec![1u8, 0]));
}

#[test]
fn a_custom_collator_receives_whole_samples_and_shapes_the_output_type() {
    /// Sums each sample into one value per slot, proving collation sees
    /// samples rather than pre-flattened data.
    struct SumCollate;
    impl Collate<i32> for SumCollate {
        type Output = i64;
        fn collate(&self, batch: Vec<i32>) -> BatchResult<Self::Output> {
            Ok(batch.iter().map(|&v| i64::from(v)).sum())
        }
    }

    let loader = DataLoader::builder_with_collate(Squares::new(5), SumCollate)
        .batch_size(3)
        .build()
        .expect("valid configuration");
    let sums = drain(&loader);

    assert_eq!(sums, vec![5, 25]);
}

#[test]
fn transform_composition_applies_each_step_in_order() {
    // A zero-probability flip is a deterministic identity, so the crop
    // receives the original layout and the composition is checkable exactly.
    let pipeline = incin_data::Compose::new()
        .push(incin_data::RandomHorizontalFlip::new(0.0))
        .push(incin_data::CenterCrop::new(2, 2));

    let data: Vec<f32> = (0..16).map(|v| v as f32).collect();
    let (out, shape) = pipeline
        .transform((data, vec![1, 4, 4]))
        .expect("a 4x4 image crops to 2x2");

    assert_eq!(shape, vec![1, 2, 2]);
    assert_eq!(out, vec![5.0, 6.0, 9.0, 10.0]);
}

#[test]
fn a_failing_transform_is_a_typed_invalid_input_error() {
    let pipeline = incin_data::Compose::new().push(incin_data::CenterCrop::new(4, 4));
    let error = pipeline
        .transform((vec![0.0; 16], vec![1, 2, 2]))
        .expect_err("cropping 2x2 to 4x4 has no center");

    match error {
        DataError::InvalidInput(message) => {
            assert!(
                message.contains("exceed image dimensions"),
                "unexpected: {message}"
            );
        }
        other => panic!("expected DataError::InvalidInput, got {other:?}"),
    }
}
