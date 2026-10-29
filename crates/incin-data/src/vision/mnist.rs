use crate::loader::{BatchResult, Collate, DataError};
#[cfg(not(feature = "download"))]
use incin_core::error::ErrorMessage;
use incin_core::error::{Error, Result};
use std::path::Path;

/// Mnist dataset.
pub struct MnistDataset {
    images: Vec<u8>,
    labels: Vec<u8>,
    train: bool,
}

/// Converts an MNIST batch to model-ready tensors on an explicitly selected target.
///
/// The data crate owns this small adapter contract without depending on any
/// concrete backend crate. The facade implements it for `TensorTarget` values.
pub trait MnistBatchTarget: Clone + Send + Sync + 'static {
    /// Image batch type.
    type Images: Send + 'static;
    /// Label batch type.
    type Labels: Send + 'static;

    /// Builds normalized images and integer labels on this target.
    fn batch(
        &self,
        images: Vec<f32>,
        labels: Vec<u8>,
        batch_size: usize,
    ) -> BatchResult<(Self::Images, Self::Labels)>;
}

/// Batches MNIST samples directly into model-ready tensors for target `T`.
///
/// Images are normalized to `f32` with shape `[batch, 1, 28, 28]`. Labels
/// remain integer `u8` values and carry no gradient marker, so data loading does not
/// create a training graph. The target value is the explicit device choice.
#[derive(Debug, Clone, Copy, Default)]
pub struct TensorCollate<T>(T);

impl<T> TensorCollate<T> {
    /// Creates a target-aware tensor batcher for `target`.
    #[must_use]
    pub const fn new(target: T) -> Self {
        Self(target)
    }
}

impl<T> Collate<(Vec<f32>, u8)> for TensorCollate<T>
where
    T: MnistBatchTarget,
{
    type Output = (T::Images, T::Labels);

    fn collate(&self, batch: Vec<(Vec<f32>, u8)>) -> BatchResult<Self::Output> {
        const PIXELS_PER_IMAGE: usize = 28 * 28;
        let batch_size = batch.len();
        let mut images = Vec::with_capacity(batch_size * PIXELS_PER_IMAGE);
        let mut labels = Vec::with_capacity(batch_size);

        for (image, label) in batch {
            if image.len() != PIXELS_PER_IMAGE {
                return Err(DataError::InvalidBatch(format!(
                    "MNIST image has {} values, expected {PIXELS_PER_IMAGE}",
                    image.len()
                )));
            }
            images.extend(image);
            labels.push(label);
        }

        self.0.batch(images, labels, batch_size)
    }
}

impl MnistDataset {
    /// Starts a model-ready loader using `target` as the explicit target.
    ///
    /// The returned builder uses [`TensorCollate`] internally, so ordinary
    /// MNIST application code does not need to assemble host vectors or name
    /// a custom collator.
    #[must_use]
    pub fn loader<T>(self, target: T) -> crate::loader::DataLoaderBuilder<Self, TensorCollate<T>>
    where
        T: MnistBatchTarget,
    {
        crate::loader::DataLoader::builder_with_collate(self, TensorCollate::new(target))
    }

    /// New.
    pub fn new<P: AsRef<Path>>(dir: P, train: bool) -> Result<Self> {
        let dir = dir.as_ref();
        let (images_url, labels_url) = if train {
            (
                "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
                "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
            )
        } else {
            (
                "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
                "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
            )
        };

        let images_archive = if train {
            "train-images-idx3-ubyte.gz"
        } else {
            "t10k-images-idx3-ubyte.gz"
        };
        let labels_archive = if train {
            "train-labels-idx1-ubyte.gz"
        } else {
            "t10k-labels-idx1-ubyte.gz"
        };
        // The archive names above are compile-time constants carrying the
        // suffix; a mismatch would be an internal contradiction, not caller
        // input.
        let images_name =
            images_archive
                .strip_suffix(".gz")
                .ok_or_else(|| Error::InternalInvariant {
                    operation: "load mnist",
                    reason: "image archive name is not gzip-compressed",
                })?;
        let labels_name =
            labels_archive
                .strip_suffix(".gz")
                .ok_or_else(|| Error::InternalInvariant {
                    operation: "load mnist",
                    reason: "label archive name is not gzip-compressed",
                })?;

        std::fs::create_dir_all(dir).map_err(|e| super::idx::io_error("load mnist", e))?;

        // Fetching the archives is the `download` feature's job. Without it the
        // rest of this function still works against files already on disk, so
        // the refusal names the missing step rather than the missing module.
        #[cfg(feature = "download")]
        {
            crate::downloader::Downloader::download_and_extract_gz(images_url, dir, images_name)?;
            crate::downloader::Downloader::download_and_extract_gz(labels_url, dir, labels_name)?;
        }
        #[cfg(not(feature = "download"))]
        {
            let _ = (images_url, labels_url);
            if !dir.join(images_name).is_file() || !dir.join(labels_name).is_file() {
                return Err(Error::Io {
                    operation: "load mnist",
                    message: ErrorMessage::new(format!(
                        "MNIST archives are not present in {} and this build cannot fetch them: \
                         enable the `download` feature of incin-data, or extract \
                         {images_name} and {labels_name} into that directory yourself",
                        dir.display()
                    )),
                });
            }
        }

        let images_file = dir.join(images_name);
        let labels_file = dir.join(labels_name);

        let images = Self::parse_images(&images_file)?;
        let labels = Self::parse_labels(&labels_file)?;

        Self::try_from_parts(images, labels, train)
    }

    fn try_from_parts(images: Vec<u8>, labels: Vec<u8>, train: bool) -> Result<Self> {
        const PIXELS_PER_IMAGE: usize = 28 * 28;
        super::idx::check_image_label_parts(&images, &labels, PIXELS_PER_IMAGE, 9, "load mnist")?;
        Ok(Self {
            images,
            labels,
            train,
        })
    }

    /// Returns whether this is the training split.
    #[must_use]
    pub const fn is_training(&self) -> bool {
        self.train
    }

    /// Returns the validated image bytes.
    #[must_use]
    pub fn image_bytes(&self) -> &[u8] {
        &self.images
    }

    /// Returns the validated labels.
    #[must_use]
    pub fn labels(&self) -> &[u8] {
        &self.labels
    }

    /// Parse images.
    fn parse_images(path: &Path) -> Result<Vec<u8>> {
        super::idx::read_idx_images(path, 28, 28, 100_000, 1000, "load mnist")
    }

    /// Parse labels.
    fn parse_labels(path: &Path) -> Result<Vec<u8>> {
        super::idx::read_idx_labels(path, 9, 100_000, "load mnist")
    }
}

impl crate::dataset::Dataset for MnistDataset {
    /// Item.
    type Item = (Vec<f32>, u8);

    /// Len.
    fn len(&self) -> usize {
        self.labels.len()
    }

    /// Get.
    fn get(
        &self,
        index: usize,
    ) -> core::result::Result<Option<Self::Item>, crate::loader::DataError> {
        if index >= self.labels.len() {
            return Ok(None);
        }
        let label = self.labels[index];
        const PIXELS_PER_IMAGE: usize = 28 * 28;
        let img = super::idx::image_window(&self.images, index, PIXELS_PER_IMAGE, "MNIST")?;
        let mut img_f32 = Vec::with_capacity(28 * 28);
        for &b in img {
            img_f32.push(b as f32 / 255.0);
        }
        Ok(Some((img_f32, label)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;
    #[derive(Clone)]
    struct TestTarget;

    impl MnistBatchTarget for TestTarget {
        type Images = (Vec<f32>, Vec<usize>);
        type Labels = (Vec<u8>, Vec<usize>);

        fn batch(
            &self,
            images: Vec<f32>,
            labels: Vec<u8>,
            batch_size: usize,
        ) -> BatchResult<(Self::Images, Self::Labels)> {
            Ok((
                (images, vec![batch_size, 1, 28, 28]),
                (labels, vec![batch_size]),
            ))
        }
    }

    #[test]
    fn validated_parts_reject_mismatched_image_and_label_counts() {
        assert!(MnistDataset::try_from_parts(vec![0; 28 * 28], vec![0, 1], true).is_err());
    }

    #[test]
    fn validated_parts_reject_non_digit_labels() {
        assert!(MnistDataset::try_from_parts(vec![0; 28 * 28], vec![10], true).is_err());
    }

    #[test]
    fn validated_parts_preserve_split_and_checked_indexing() {
        let dataset = MnistDataset::try_from_parts(vec![255; 2 * 28 * 28], vec![3, 7], false)
            .expect("matching image and label counts should construct");
        assert!(!dataset.is_training());
        assert_eq!(dataset.image_bytes().len(), 2 * 28 * 28);
        assert_eq!(dataset.labels(), &[3, 7]);
        assert_eq!(dataset.get(1).unwrap().unwrap().1, 7);
        assert!(dataset.get(2).unwrap().is_none());
    }

    #[test]
    fn tensor_collate_returns_targeted_no_grad_batches() {
        let batch = TensorCollate::new(TestTarget)
            .collate(vec![(vec![0.25; 28 * 28], 3), (vec![0.5; 28 * 28], 7)])
            .expect("valid MNIST samples should form a tensor batch");

        assert_eq!(batch.0.1, vec![2, 1, 28, 28]);
        assert_eq!(batch.1.1, vec![2]);
    }

    #[test]
    fn tensor_collate_rejects_incompatible_image_shape() {
        let error = TensorCollate::new(TestTarget)
            .collate(vec![(vec![0.0; 28 * 28 - 1], 0)])
            .expect_err("a malformed image must not enter a model batch");

        assert!(matches!(error, DataError::InvalidBatch(_)));
    }

    #[test]
    fn parse_images_rejects_wrong_magic_as_malformed_artifact() {
        let path = std::env::temp_dir().join(format!(
            "incin-mnist-bad-images-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock is after the epoch")
                .as_nanos()
        ));
        std::fs::write(&path, [0u8, 0, 0, 99, 0, 0, 0, 1, 0, 0, 0, 28, 0, 0, 0, 28])
            .expect("temporary IDX file should write");
        let error = MnistDataset::parse_images(&path)
            .expect_err("a wrong magic number is not an image archive");
        let _ = std::fs::remove_file(&path);
        assert!(matches!(
            error,
            Error::MalformedArtifact {
                operation: "load mnist",
                artifact: "image archive",
                ..
            }
        ));
    }

    #[test]
    fn parse_labels_reject_oversized_counts_as_resource_limit() {
        let path = std::env::temp_dir().join(format!(
            "incin-mnist-large-labels-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock is after the epoch")
                .as_nanos()
        ));
        // Magic 2049 with a count above the 100_000 bound; the body is
        // irrelevant because the header is refused before it is read.
        let mut header = Vec::new();
        header.extend_from_slice(&2049u32.to_be_bytes());
        header.extend_from_slice(&200_000u32.to_be_bytes());
        header.extend_from_slice(&[0u8; 4]);
        std::fs::write(&path, header).expect("temporary IDX file should write");
        let error = MnistDataset::parse_labels(&path)
            .expect_err("a header count past the limit is refused");
        let _ = std::fs::remove_file(&path);
        assert!(matches!(
            error,
            Error::ResourceLimit {
                operation: "load mnist",
                resource: "label count",
                actual: 200_000,
                limit: 100_000,
            }
        ));
    }
}
