use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Mnist dataset.
pub struct MnistDataset {
    /// Images.
    pub images: Vec<u8>,
    /// Labels.
    pub labels: Vec<u8>,
    /// Train.
    pub train: bool,
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

        Ok(Self {
            images,
            labels,
            train,
        })
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
            return Err(anyhow::anyhow!("Invalid IDX magic number for MNIST images: expected 2051, got {}", magic_val));
        }

        let count = u32::from_be_bytes(count) as usize;
        let rows = u32::from_be_bytes(rows) as usize;
        let cols = u32::from_be_bytes(cols) as usize;

        if count > 100_000 || rows > 1000 || cols > 1000 {
            return Err(anyhow::anyhow!("MNIST image header parameters exceed safe resource limits: count={}, rows={}, cols={}", count, rows, cols));
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
            return Err(anyhow::anyhow!("Invalid IDX magic number for MNIST labels: expected 2049, got {}", magic_val));
        }

        let count = u32::from_be_bytes(count) as usize;
        if count > 100_000 {
            return Err(anyhow::anyhow!("MNIST label count exceeds safe resource limit: count={}", count));
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
        let start = index * 28 * 28;
        let end = start + 28 * 28;
        let img = &self.images[start..end];
        let mut img_f32 = Vec::with_capacity(28 * 28);
        for &b in img {
            img_f32.push(b as f32 / 255.0);
        }
        Some((img_f32, label))
    }
}
