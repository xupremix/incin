//! Fashion-MNIST: 28x28 grayscale clothing articles, IDX archives with
//! the same layout as MNIST (magic 2051/2049) under different filenames.
//! Parsing, validation windows and batching ride the shared IDX readers
//! and MNIST collation; this file only names its own archives.

use super::idx;
use super::mnist::{MnistBatchTarget, TensorCollate};
use incin_core::error::{Error, Result};
use std::path::Path;

/// Fashion-MNIST dataset.
pub struct FashionMnistDataset {
    images: Vec<u8>,
    labels: Vec<u8>,
    train: bool,
}

impl FashionMnistDataset {
    /// Starts a model-ready loader using `target` as the explicit target.
    ///
    /// Images are `[batch, 1, 28, 28]` f32 like MNIST; only the archive
    /// contents differ.
    #[must_use]
    pub fn loader<T>(self, target: T) -> crate::loader::DataLoaderBuilder<Self, TensorCollate<T>>
    where
        T: MnistBatchTarget,
    {
        crate::loader::DataLoader::builder_with_collate(self, TensorCollate::new(target))
    }

    /// New.
    pub fn new<P: AsRef<Path>>(dir: P, train: bool) -> Result<Self> {
        const OPERATION: &str = "load fashion-mnist";
        let dir = dir.as_ref();
        let (images_url, labels_url) = if train {
            (
                "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz",
                "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz",
            )
        } else {
            (
                "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
                "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
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
        let images_name =
            images_archive
                .strip_suffix(".gz")
                .ok_or_else(|| Error::InternalInvariant {
                    operation: OPERATION,
                    reason: "image archive name is not gzip-compressed",
                })?;
        let labels_name =
            labels_archive
                .strip_suffix(".gz")
                .ok_or_else(|| Error::InternalInvariant {
                    operation: OPERATION,
                    reason: "label archive name is not gzip-compressed",
                })?;

        std::fs::create_dir_all(dir).map_err(|e| idx::io_error(OPERATION, e))?;

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
                    operation: OPERATION,
                    message: ErrorMessage::new(format!(
                        "Fashion-MNIST archives are not present in {} and this build cannot fetch them: \
                         enable the `download` feature of incin-data, or extract \
                         {images_name} and {labels_name} into that directory yourself",
                        dir.display()
                    )),
                });
            }
        }

        let images =
            idx::read_idx_images(&dir.join(images_name), 28, 28, 100_000, 1000, OPERATION)?;
        let labels = idx::read_idx_labels(&dir.join(labels_name), 9, 100_000, OPERATION)?;

        Self::try_from_parts(images, labels, train)
    }

    fn try_from_parts(images: Vec<u8>, labels: Vec<u8>, train: bool) -> Result<Self> {
        const PIXELS_PER_IMAGE: usize = 28 * 28;
        idx::check_image_label_parts(&images, &labels, PIXELS_PER_IMAGE, 9, Self::OPERATION)?;
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

    const OPERATION: &'static str = "load fashion-mnist";
}

impl crate::dataset::Dataset for FashionMnistDataset {
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
        let img = idx::image_window(&self.images, index, PIXELS_PER_IMAGE, "Fashion-MNIST")?;
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

    #[test]
    fn validated_parts_reject_mismatched_counts() {
        assert!(FashionMnistDataset::try_from_parts(vec![0; 28 * 28], vec![0, 1], true).is_err());
    }

    #[test]
    fn validated_parts_preserve_split_and_indexing() {
        let dataset =
            FashionMnistDataset::try_from_parts(vec![128; 2 * 28 * 28], vec![2, 9], false)
                .expect("matching parts should construct");
        assert!(!dataset.is_training());
        assert_eq!(dataset.labels(), &[2, 9]);
        let (img, label) = dataset.get(0).unwrap().unwrap();
        assert_eq!(label, 2);
        assert_eq!(img.len(), 28 * 28);
        assert!((img[0] - 128.0 / 255.0).abs() < 1e-6);
        assert!(dataset.get(2).unwrap().is_none());
    }

    #[test]
    fn idx_roundtrip_through_real_headers() {
        // A minimal synthetic IDX pair: magic, count, geometry, body.
        let dir = std::env::temp_dir().join(format!(
            "incin-fashion-idx-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock is after the epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir should create");
        let mut images = Vec::new();
        images.extend_from_slice(&2051u32.to_be_bytes());
        images.extend_from_slice(&2u32.to_be_bytes());
        images.extend_from_slice(&28u32.to_be_bytes());
        images.extend_from_slice(&28u32.to_be_bytes());
        images.extend_from_slice(&[7u8; 2 * 28 * 28]);
        std::fs::write(dir.join("train-images-idx3-ubyte"), &images).expect("write images");
        let mut labels = Vec::new();
        labels.extend_from_slice(&2049u32.to_be_bytes());
        labels.extend_from_slice(&2u32.to_be_bytes());
        labels.extend_from_slice(&[4u8, 5u8]);
        std::fs::write(dir.join("train-labels-idx1-ubyte"), &labels).expect("write labels");
        let dataset = FashionMnistDataset::new(&dir, true).expect("synthetic pair should load");
        assert!(dataset.is_training());
        assert_eq!(dataset.labels(), &[4, 5]);
        assert_eq!(dataset.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
