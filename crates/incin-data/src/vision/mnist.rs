use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::loader::{BatchResult, Collate, DataError};

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
    pub fn new<P: AsRef<Path>>(dir: P, train: bool) -> anyhow::Result<Self> {
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
        let images_name = images_archive
            .strip_suffix(".gz")
            .ok_or_else(|| anyhow::anyhow!("MNIST image archive is not gzip-compressed"))?;
        let labels_name = labels_archive
            .strip_suffix(".gz")
            .ok_or_else(|| anyhow::anyhow!("MNIST label archive is not gzip-compressed"))?;

        std::fs::create_dir_all(dir)?;

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
                return Err(anyhow::anyhow!(
                    "MNIST archives are not present in {} and this build cannot fetch them: \
                     enable the `download` feature of incin-data, or extract \
                     {images_name} and {labels_name} into that directory yourself",
                    dir.display()
                ));
            }
        }

        let images_file = dir.join(images_name);
        let labels_file = dir.join(labels_name);

        let images = Self::parse_images(&images_file)?;
        let labels = Self::parse_labels(&labels_file)?;

        Self::try_from_parts(images, labels, train)
    }

    fn try_from_parts(images: Vec<u8>, labels: Vec<u8>, train: bool) -> anyhow::Result<Self> {
        const PIXELS_PER_IMAGE: usize = 28 * 28;
        if labels.iter().any(|&label| label > 9) {
            return Err(anyhow::anyhow!(
                "MNIST labels must be decimal digits in the range 0..=9"
            ));
        }
        let expected_images = labels
            .len()
            .checked_mul(PIXELS_PER_IMAGE)
            .ok_or_else(|| anyhow::anyhow!("MNIST image count overflows address space"))?;
        if images.len() != expected_images {
            return Err(anyhow::anyhow!(
                "MNIST images/labels mismatch: {} image bytes for {} labels",
                images.len(),
                labels.len()
            ));
        }
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
    fn parse_images(path: &Path) -> anyhow::Result<Vec<u8>> {
        let mut f = File::open(path)?;
        let mut magic = [0u8; 4];
        let mut count = [0u8; 4];
        let mut rows = [0u8; 4];
        let mut cols = [0u8; 4];

        f.read_exact(&mut magic)?;
        f.read_exact(&mut count)?;
        f.read_exact(&mut rows)?;
        f.read_exact(&mut cols)?;

        let magic_val = u32::from_be_bytes(magic);
        if magic_val != 2051 {
            return Err(anyhow::anyhow!(
                "Invalid IDX magic number for MNIST images: expected 2051, got {}",
                magic_val
            ));
        }

        let count = usize::try_from(u32::from_be_bytes(count))?;
        let rows = usize::try_from(u32::from_be_bytes(rows))?;
        let cols = usize::try_from(u32::from_be_bytes(cols))?;

        if count > 100_000 || rows > 1000 || cols > 1000 {
            return Err(anyhow::anyhow!(
                "MNIST image header parameters exceed safe resource limits: count={}, rows={}, cols={}",
                count,
                rows,
                cols
            ));
        }
        if rows != 28 || cols != 28 {
            return Err(anyhow::anyhow!(
                "MNIST images must be 28x28, got {}x{}",
                rows,
                cols
            ));
        }

        let num_bytes = count
            .checked_mul(rows)
            .and_then(|v| v.checked_mul(cols))
            .ok_or_else(|| anyhow::anyhow!("Arithmetic overflow in MNIST image data size"))?;

        let mut data = vec![0u8; num_bytes];
        f.read_exact(&mut data)?;
        if f.read(&mut [0u8; 1])? != 0 {
            return Err(anyhow::anyhow!("MNIST image file contains trailing bytes"));
        }

        Ok(data)
    }

    /// Parse labels.
    fn parse_labels(path: &Path) -> anyhow::Result<Vec<u8>> {
        let mut f = File::open(path)?;
        let mut magic = [0u8; 4];
        let mut count = [0u8; 4];

        f.read_exact(&mut magic)?;
        f.read_exact(&mut count)?;

        let magic_val = u32::from_be_bytes(magic);
        if magic_val != 2049 {
            return Err(anyhow::anyhow!(
                "Invalid IDX magic number for MNIST labels: expected 2049, got {}",
                magic_val
            ));
        }

        let count = usize::try_from(u32::from_be_bytes(count))?;
        if count > 100_000 {
            return Err(anyhow::anyhow!(
                "MNIST label count exceeds safe resource limit: count={}",
                count
            ));
        }

        let mut data = vec![0u8; count];
        f.read_exact(&mut data)?;
        if data.iter().any(|&label| label > 9) {
            return Err(anyhow::anyhow!(
                "MNIST labels must be decimal digits in the range 0..=9"
            ));
        }
        if f.read(&mut [0u8; 1])? != 0 {
            return Err(anyhow::anyhow!("MNIST label file contains trailing bytes"));
        }

        Ok(data)
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
    fn get(&self, index: usize) -> Result<Option<Self::Item>, crate::loader::DataError> {
        if index >= self.labels.len() {
            return Ok(None);
        }
        let label = self.labels[index];
        const PIXELS_PER_IMAGE: usize = 28 * 28;
        let start = index.checked_mul(PIXELS_PER_IMAGE).ok_or_else(|| {
            crate::loader::DataError::Dataset("MNIST image offset overflow".into())
        })?;
        let end = start.checked_add(PIXELS_PER_IMAGE).ok_or_else(|| {
            crate::loader::DataError::Dataset("MNIST image end offset overflow".into())
        })?;
        let img = self.images.get(start..end).ok_or_else(|| {
            crate::loader::DataError::Dataset("MNIST image buffer is truncated".into())
        })?;
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
}
