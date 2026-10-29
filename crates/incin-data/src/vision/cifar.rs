//! CIFAR-10 and CIFAR-100: 32x32 RGB images as flat binary records
//! (no IDX headers, unlike MNIST).
//!
//! CIFAR-10 records are 3073 bytes (`label, R(1024), G(1024), B(1024)`);
//! CIFAR-100 records are 3074 bytes (`coarse, fine, R, G, B`) and only the
//! fine label is exposed - the coarse label is a 20-way grouping most
//! training loops never name. Train reads the five `data_batch_*.bin`
//! files (CIFAR-10) or `train.bin` (CIFAR-100); test reads
//! `test_batch.bin` / `test.bin`, all inside the directory the official
//! tarballs extract to. There is deliberately no download support: the
//! archives are `.tar.gz` and the downloader only fetches single `.gz`
//! members, so `new` names the extracted files it needs instead of
//! pretending to fetch them.
//!
//! Items are `(Vec<f32>, u8)` like MNIST: 3072 channel-major floats in
//! `[0, 1]`, so the same target adapters batch all three catalog datasets.

use super::mnist::{MnistBatchTarget, TensorCollate};
use crate::loader::DataError;
use incin_core::error::{Error, ErrorMessage, Result};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Pixels per CIFAR image, channel-major.
const PIXELS_PER_IMAGE: usize = 3 * 32 * 32;
/// Hard cap on records per split (CIFAR-100 train is 50k).
const MAX_RECORDS: usize = 100_000;

/// One parsed split: flat channel-major bytes plus fine labels.
struct Split {
    images: Vec<u8>,
    labels: Vec<u8>,
}

fn io_error(operation: &'static str, source: std::io::Error) -> Error {
    Error::Io {
        operation,
        message: ErrorMessage::new(source.to_string()),
    }
}

fn malformed(operation: &'static str, artifact: &'static str, reason: impl AsRef<str>) -> Error {
    Error::MalformedArtifact {
        operation,
        artifact,
        reason: ErrorMessage::new(reason),
    }
}

/// Reads whole files and splits them into fixed-size records, refusing
/// truncation and trailing bytes per file.
fn read_records(
    dir: &Path,
    files: &[&str],
    record_len: usize,
    label_at: usize,
    max_label: u8,
    operation: &'static str,
) -> Result<Split> {
    let mut images = Vec::new();
    let mut labels = Vec::new();
    for file in files {
        let path = dir.join(file);
        let mut f = File::open(&path).map_err(|e| Error::Io {
            operation,
            message: ErrorMessage::new(format!(
                "CIFAR split file {} is not readable in {} ({e}): download and extract the \
                     official binary tarball into that directory yourself",
                path.display(),
                dir.display()
            )),
        })?;
        let mut data = Vec::new();
        f.read_to_end(&mut data)
            .map_err(|e| io_error(operation, e))?;
        if data.len() % record_len != 0 {
            return Err(malformed(
                operation,
                "split file",
                format!(
                    "{file} has {} bytes, not a multiple of the {record_len}-byte record",
                    data.len()
                ),
            ));
        }
        let records = data.len() / record_len;
        if labels.len() + records > MAX_RECORDS {
            return Err(Error::ResourceLimit {
                operation,
                resource: "record count",
                actual: (labels.len() + records) as u64,
                limit: MAX_RECORDS as u64,
            });
        }
        for chunk in data.chunks_exact(record_len) {
            let label = chunk[label_at];
            if label > max_label {
                return Err(malformed(
                    operation,
                    "split file",
                    format!("label {label} exceeds {max_label} in {file}"),
                ));
            }
            labels.push(label);
            images.extend_from_slice(&chunk[label_at + 1..]);
        }
    }
    Ok(Split { images, labels })
}

/// CIFAR-10 dataset (10 classes, 50k train / 10k test).
pub struct Cifar10Dataset {
    images: Vec<u8>,
    labels: Vec<u8>,
    train: bool,
}

/// CIFAR-100 dataset (100 fine classes, 50k train / 10k test).
pub struct Cifar100Dataset {
    images: Vec<u8>,
    labels: Vec<u8>,
    train: bool,
}

macro_rules! cifar_dataset {
    ($name:ident, $op:literal, $train_files:expr, $test_file:expr, $record_len:expr, $label_at:expr, $max_label:expr, $url:literal, $tarball:literal, $topdir:literal) => {
        impl $name {
            /// Starts a model-ready loader using `target` as the explicit target.
            ///
            /// Images are `[batch, 3, 32, 32]` f32 in `[0, 1]`.
            #[must_use]
            pub fn loader<T>(
                self,
                target: T,
            ) -> crate::loader::DataLoaderBuilder<Self, TensorCollate<T>>
            where
                T: MnistBatchTarget,
            {
                crate::loader::DataLoader::builder_with_collate(self, TensorCollate::new(target))
            }

            /// Reads the extracted binary tarball from `dir` (see the module
            /// docs for the download URL).
            pub fn new<P: AsRef<Path>>(dir: P, train: bool) -> Result<Self> {
                let dir = dir.as_ref();
                let wanted: &[&str] = if train { &$train_files } else { &[$test_file] };
                // Files straight in `dir` win: tests and pre-extracted
                // trees never touch the network. Otherwise, with the
                // `download` feature, fetch the tarball once and read
                // from inside its top directory.
                let base: std::path::PathBuf;
                if wanted.iter().all(|f| dir.join(f).is_file()) {
                    base = dir.to_path_buf();
                } else {
                    #[cfg(feature = "download")]
                    {
                        base = crate::downloader::Downloader::download_and_extract_tar_gz(
                            $url, dir, $tarball, $topdir,
                        )?;
                    }
                    #[cfg(not(feature = "download"))]
                    {
                        return Err(Error::Io {
                            operation: $op,
                            message: ErrorMessage::new(format!(
                                "CIFAR split files are missing from {} and this build cannot fetch \
                                 them: enable the `download` feature of incin-data, or extract the \
                                 official binary tarball into that directory yourself",
                                dir.display()
                            )),
                        });
                    }
                }
                let split = read_records(&base, wanted, $record_len, $label_at, $max_label, $op)?;
                Ok(Self {
                    images: split.images,
                    labels: split.labels,
                    train,
                })
            }

            /// Returns whether this is the training split.
            #[must_use]
            pub const fn is_training(&self) -> bool {
                self.train
            }

            /// Returns the validated image bytes, channel-major.
            #[must_use]
            pub fn image_bytes(&self) -> &[u8] {
                &self.images
            }

            /// Returns the validated fine labels.
            #[must_use]
            pub fn labels(&self) -> &[u8] {
                &self.labels
            }
        }

        impl crate::dataset::Dataset for $name {
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
                let img = super::idx::image_window(&self.images, index, PIXELS_PER_IMAGE, $op)?;
                let mut img_f32 = Vec::with_capacity(PIXELS_PER_IMAGE);
                for &b in img {
                    img_f32.push(b as f32 / 255.0);
                }
                Ok(Some((img_f32, label)))
            }
        }
    };
}

cifar_dataset!(
    Cifar10Dataset,
    "load cifar-10",
    [
        "data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin"
    ],
    "test_batch.bin",
    3073,
    0,
    9,
    "https://cave.cs.toronto.edu/kriz/cifar-10-binary.tar.gz",
    "cifar-10-binary.tar.gz",
    "cifar-10-batches-bin"
);

cifar_dataset!(
    Cifar100Dataset,
    "load cifar-100",
    ["train.bin"],
    "test.bin",
    3074,
    1,
    99,
    "https://cave.cs.toronto.edu/kriz/cifar-100-binary.tar.gz",
    "cifar-100-binary.tar.gz",
    "cifar-100-binary"
);

/// `TensorCollate` serves MNIST-shaped batches only; CIFAR batches stack
/// `[3, 32, 32]` images through the same target contract.
pub struct CifarCollate<T>(T);

impl<T> CifarCollate<T> {
    /// Creates a target-aware tensor batcher for `target`.
    #[must_use]
    pub const fn new(target: T) -> Self {
        Self(target)
    }
}

impl<T> crate::loader::Collate<(Vec<f32>, u8)> for CifarCollate<T>
where
    T: MnistBatchTarget,
{
    type Output = (T::Images, T::Labels);

    fn collate(&self, batch: Vec<(Vec<f32>, u8)>) -> crate::loader::BatchResult<Self::Output> {
        let batch_size = batch.len();
        let mut images = Vec::with_capacity(batch_size * PIXELS_PER_IMAGE);
        let mut labels = Vec::with_capacity(batch_size);

        for (image, label) in batch {
            if image.len() != PIXELS_PER_IMAGE {
                return Err(DataError::InvalidBatch(format!(
                    "CIFAR image has {} values, expected {PIXELS_PER_IMAGE}",
                    image.len()
                )));
            }
            images.extend(image);
            labels.push(label);
        }

        self.0.batch(images, labels, batch_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;
    use crate::loader::Collate;

    fn cifar10_record(label: u8, fill: u8) -> Vec<u8> {
        let mut record = vec![label];
        record.extend_from_slice(&[fill; 3072]);
        record
    }

    fn write_split(dir: &Path, name: &str, records: &[Vec<u8>]) {
        let mut data = Vec::new();
        for record in records {
            data.extend_from_slice(record);
        }
        std::fs::write(dir.join(name), data).expect("synthetic split should write");
    }

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "incin-{prefix}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock is after the epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir should create");
        dir
    }

    #[test]
    fn cifar10_concatenates_five_train_batches() {
        let dir = temp_dir("cifar10");
        for batch in 1..=5 {
            write_split(
                &dir,
                &format!("data_batch_{batch}.bin"),
                &[cifar10_record(batch as u8, batch as u8 * 10)],
            );
        }
        let dataset = Cifar10Dataset::new(&dir, true).expect("five synthetic batches should load");
        assert!(dataset.is_training());
        assert_eq!(dataset.len(), 5);
        assert_eq!(dataset.labels(), &[1, 2, 3, 4, 5]);
        // Channel-major RGB: the first record's R/G/B planes start at
        // 0/1024/2048 within its 3072 bytes.
        let (img, label) = dataset.get(0).unwrap().unwrap();
        assert_eq!(label, 1);
        assert_eq!(img.len(), 3072);
        assert!((img[0] - 10.0 / 255.0).abs() < 1e-6);
        assert!((img[1024] - 10.0 / 255.0).abs() < 1e-6);
        assert!((img[2048] - 10.0 / 255.0).abs() < 1e-6);
        assert!(dataset.get(5).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cifar10_truncated_file_is_malformed() {
        let dir = temp_dir("cifar10-trunc");
        std::fs::write(dir.join("test_batch.bin"), vec![3u8; 100]).expect("write");
        let error = Cifar10Dataset::new(&dir, false);
        assert!(
            matches!(error, Err(Error::MalformedArtifact { .. })),
            "a truncated record is not a split"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cifar100_reads_fine_labels() {
        let dir = temp_dir("cifar100");
        let mut train = Vec::new();
        // coarse 7, fine 42, pixels 9.
        let mut record = vec![7u8, 42u8];
        record.extend_from_slice(&[9u8; 3072]);
        train.extend_from_slice(&record);
        std::fs::write(dir.join("train.bin"), train).expect("write");
        let dataset = Cifar100Dataset::new(&dir, true).expect("synthetic pair should load");
        assert_eq!(dataset.labels(), &[42]);
        let (img, label) = dataset.get(0).unwrap().unwrap();
        assert_eq!(label, 42);
        assert!((img[0] - 9.0 / 255.0).abs() < 1e-6);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cifar_collate_stacks_rgb_batches() {
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
            ) -> crate::loader::BatchResult<(Self::Images, Self::Labels)> {
                Ok((
                    (images, vec![batch_size, 3, 32, 32]),
                    (labels, vec![batch_size]),
                ))
            }
        }
        let batch = CifarCollate::new(TestTarget)
            .collate(vec![(vec![0.5; 3072], 3), (vec![0.25; 3072], 7)])
            .expect("valid CIFAR samples should batch");
        assert_eq!(batch.0.1, vec![2, 3, 32, 32]);
        assert_eq!(batch.1.1, vec![2]);
    }
}
