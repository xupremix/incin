use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Mnist dataset.
pub struct MnistDataset {
    images: Vec<u8>,
    labels: Vec<u8>,
    train: bool,
}

impl MnistDataset {
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

        let images_file = dir.join(images_url.split('/').next_back().unwrap());
        let labels_file = dir.join(labels_url.split('/').next_back().unwrap());

        std::fs::create_dir_all(dir)?;

        crate::downloader::Downloader::download_and_extract_gz(
            images_url,
            dir,
            images_file
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .strip_suffix(".gz")
                .unwrap_or(images_file.file_name().unwrap().to_str().unwrap()),
        )?;
        crate::downloader::Downloader::download_and_extract_gz(
            labels_url,
            dir,
            labels_file
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .strip_suffix(".gz")
                .unwrap_or(labels_file.file_name().unwrap().to_str().unwrap()),
        )?;

        // Adjust the parsed file path to the extracted file, without .gz
        let images_file = dir.join(
            images_file
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .strip_suffix(".gz")
                .unwrap(),
        );
        let labels_file = dir.join(
            labels_file
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .strip_suffix(".gz")
                .unwrap(),
        );

        let images = Self::parse_images(&images_file)?;
        let labels = Self::parse_labels(&labels_file)?;

        Self::try_from_parts(images, labels, train)
    }

    fn try_from_parts(images: Vec<u8>, labels: Vec<u8>, train: bool) -> anyhow::Result<Self> {
        const PIXELS_PER_IMAGE: usize = 28 * 28;
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

        let num_bytes = count
            .checked_mul(rows)
            .and_then(|v| v.checked_mul(cols))
            .ok_or_else(|| anyhow::anyhow!("Arithmetic overflow in MNIST image data size"))?;

        let mut data = vec![0u8; num_bytes];
        f.read_exact(&mut data)?;

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
    fn get(&self, index: usize) -> Option<Self::Item> {
        if index >= self.labels.len() {
            return None;
        }
        let label = self.labels[index];
        const PIXELS_PER_IMAGE: usize = 28 * 28;
        let start = index.checked_mul(PIXELS_PER_IMAGE)?;
        let end = start.checked_add(PIXELS_PER_IMAGE)?;
        let img = self.images.get(start..end)?;
        let mut img_f32 = Vec::with_capacity(28 * 28);
        for &b in img {
            img_f32.push(b as f32 / 255.0);
        }
        Some((img_f32, label))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;

    #[test]
    fn validated_parts_reject_mismatched_image_and_label_counts() {
        assert!(MnistDataset::try_from_parts(vec![0; 28 * 28], vec![0, 1], true).is_err());
    }

    #[test]
    fn validated_parts_preserve_split_and_checked_indexing() {
        let dataset = MnistDataset::try_from_parts(vec![255; 2 * 28 * 28], vec![3, 7], false)
            .expect("matching image and label counts should construct");
        assert!(!dataset.is_training());
        assert_eq!(dataset.image_bytes().len(), 2 * 28 * 28);
        assert_eq!(dataset.labels(), &[3, 7]);
        assert_eq!(dataset.get(1).unwrap().1, 7);
        assert!(dataset.get(2).is_none());
    }
}
