use std::fs::File;
use std::io::Read;
use std::path::Path;

pub struct MnistDataset {
    pub images: Vec<u8>,
    pub labels: Vec<u8>,
    pub train: bool,
}

impl MnistDataset {
    pub fn new<P: AsRef<Path>>(dir: P, train: bool) -> anyhow::Result<Self> {
        let dir = dir.as_ref();
        let (images_url, labels_url) = if train {
            (
                "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
                "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz"
            )
        } else {
            (
                "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
                "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"
            )
        };
        
        let images_file = dir.join(images_url.split('/').last().unwrap());
        let labels_file = dir.join(labels_url.split('/').last().unwrap());
        
        std::fs::create_dir_all(dir)?;
        
        crate::downloader::Downloader::download_and_extract_gz(images_url, dir, images_file.file_name().unwrap().to_str().unwrap().strip_suffix(".gz").unwrap_or(images_file.file_name().unwrap().to_str().unwrap()))?;
        crate::downloader::Downloader::download_and_extract_gz(labels_url, dir, labels_file.file_name().unwrap().to_str().unwrap().strip_suffix(".gz").unwrap_or(labels_file.file_name().unwrap().to_str().unwrap()))?;
        
        // Adjust the parsed file path to the extracted file, without .gz
        let images_file = dir.join(images_file.file_name().unwrap().to_str().unwrap().strip_suffix(".gz").unwrap());
        let labels_file = dir.join(labels_file.file_name().unwrap().to_str().unwrap().strip_suffix(".gz").unwrap());
        
        let images = Self::parse_images(&images_file)?;
        let labels = Self::parse_labels(&labels_file)?;
        
        Ok(Self { images, labels, train })
    }
    

    
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
        
        let count = u32::from_be_bytes(count) as usize;
        let rows = u32::from_be_bytes(rows) as usize;
        let cols = u32::from_be_bytes(cols) as usize;
        
        let mut data = vec![0u8; count * rows * cols];
        f.read_exact(&mut data)?;
        
        Ok(data)
    }
    
    fn parse_labels(path: &Path) -> anyhow::Result<Vec<u8>> {
        let mut f = File::open(path)?;
        let mut magic = [0u8; 4];
        let mut count = [0u8; 4];
        
        f.read_exact(&mut magic)?;
        f.read_exact(&mut count)?;
        
        let count = u32::from_be_bytes(count) as usize;
        
        let mut data = vec![0u8; count];
        f.read_exact(&mut data)?;
        
        Ok(data)
    }
}

impl crate::dataset::Dataset for MnistDataset {
    type Item = (Vec<f32>, u8);

    fn len(&self) -> usize {
        self.labels.len()
    }

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
