//! Shared IDX archive readers for the 28x28 grayscale catalog datasets
//! (MNIST, Fashion-MNIST): big-endian headers, magic-guarded, trailing-byte
//! checked, resource-limited. One implementation so the per-dataset files
//! only name their URLs, filenames and operation strings; the refusal
//! shapes (operations, artifacts, limits) are identical by construction.

use crate::loader::DataError;
use incin_core::error::{Error, ErrorMessage, Result};
use std::fs::File;
use std::io::Read;
use std::path::Path;

/// Wraps an I/O failure with the operation that produced it.
pub(crate) fn io_error(operation: &'static str, source: std::io::Error) -> Error {
    Error::Io {
        operation,
        message: ErrorMessage::new(source.to_string()),
    }
}

/// Reports malformed archive content under the bounded artifact category.
fn malformed(operation: &'static str, artifact: &'static str, reason: impl AsRef<str>) -> Error {
    Error::MalformedArtifact {
        operation,
        artifact,
        reason: ErrorMessage::new(reason),
    }
}

/// Reads an IDX image archive: magic 2051, `[count, rows, cols]` header,
/// exactly one body of `count * rows * cols` bytes, no trailing bytes.
///
/// `expect_rows`/`expect_cols` pin the geometry callers promise downstream
/// (28x28 here); counts past `max_count` and dims past `max_dim` are
/// resource refusals, not malformed files.
pub(crate) fn read_idx_images(
    path: &Path,
    expect_rows: usize,
    expect_cols: usize,
    max_count: usize,
    max_dim: usize,
    operation: &'static str,
) -> Result<Vec<u8>> {
    let mut f = File::open(path).map_err(|e| io_error(operation, e))?;
    let mut magic = [0u8; 4];
    let mut count = [0u8; 4];
    let mut rows = [0u8; 4];
    let mut cols = [0u8; 4];

    f.read_exact(&mut magic)
        .map_err(|e| io_error(operation, e))?;
    f.read_exact(&mut count)
        .map_err(|e| io_error(operation, e))?;
    f.read_exact(&mut rows)
        .map_err(|e| io_error(operation, e))?;
    f.read_exact(&mut cols)
        .map_err(|e| io_error(operation, e))?;

    let magic_val = u32::from_be_bytes(magic);
    if magic_val != 2051 {
        return Err(malformed(
            operation,
            "image archive",
            format!("Invalid IDX magic number for images: expected 2051, got {magic_val}"),
        ));
    }

    // A u32 header field fits a usize on every target that can hold the
    // pixel data it describes; a narrower usize is an overflow, not a
    // malformed file.
    let count =
        usize::try_from(u32::from_be_bytes(count)).map_err(|_| Error::ArithmeticOverflow {
            operation,
            expression: "image header count",
        })?;
    let rows =
        usize::try_from(u32::from_be_bytes(rows)).map_err(|_| Error::ArithmeticOverflow {
            operation,
            expression: "image header rows",
        })?;
    let cols =
        usize::try_from(u32::from_be_bytes(cols)).map_err(|_| Error::ArithmeticOverflow {
            operation,
            expression: "image header cols",
        })?;
    if count > max_count {
        return Err(Error::ResourceLimit {
            operation,
            resource: "image count",
            actual: count as u64,
            limit: max_count as u64,
        });
    }
    if rows > max_dim || cols > max_dim {
        return Err(Error::ResourceLimit {
            operation,
            resource: "image dimensions",
            actual: (rows.max(cols)) as u64,
            limit: max_dim as u64,
        });
    }
    if rows != expect_rows || cols != expect_cols {
        return Err(malformed(
            operation,
            "image archive",
            format!("images must be {expect_rows}x{expect_cols}, got {rows}x{cols}"),
        ));
    }

    let num_bytes = count
        .checked_mul(rows)
        .and_then(|v| v.checked_mul(cols))
        .ok_or(Error::ArithmeticOverflow {
            operation,
            expression: "image data size",
        })?;

    let mut data = vec![0u8; num_bytes];
    f.read_exact(&mut data)
        .map_err(|e| io_error(operation, e))?;
    if f.read(&mut [0u8; 1]).map_err(|e| io_error(operation, e))? != 0 {
        return Err(malformed(
            operation,
            "image archive",
            "image file contains trailing bytes",
        ));
    }

    Ok(data)
}

/// Reads an IDX label archive: magic 2049, `[count]` header, exactly
/// `count` bytes each within `0..=max_label`, no trailing bytes.
pub(crate) fn read_idx_labels(
    path: &Path,
    max_label: u8,
    max_count: usize,
    operation: &'static str,
) -> Result<Vec<u8>> {
    let mut f = File::open(path).map_err(|e| io_error(operation, e))?;
    let mut magic = [0u8; 4];
    let mut count = [0u8; 4];

    f.read_exact(&mut magic)
        .map_err(|e| io_error(operation, e))?;
    f.read_exact(&mut count)
        .map_err(|e| io_error(operation, e))?;

    let magic_val = u32::from_be_bytes(magic);
    if magic_val != 2049 {
        return Err(malformed(
            operation,
            "label archive",
            format!("Invalid IDX magic number for labels: expected 2049, got {magic_val}"),
        ));
    }

    let count =
        usize::try_from(u32::from_be_bytes(count)).map_err(|_| Error::ArithmeticOverflow {
            operation,
            expression: "label header count",
        })?;
    if count > max_count {
        return Err(Error::ResourceLimit {
            operation,
            resource: "label count",
            actual: count as u64,
            limit: max_count as u64,
        });
    }

    let mut data = vec![0u8; count];
    f.read_exact(&mut data)
        .map_err(|e| io_error(operation, e))?;
    if data.iter().any(|&label| label > max_label) {
        return Err(malformed(
            operation,
            "label archive",
            format!("labels must be in the range 0..={max_label}"),
        ));
    }
    if f.read(&mut [0u8; 1]).map_err(|e| io_error(operation, e))? != 0 {
        return Err(malformed(
            operation,
            "label archive",
            "label file contains trailing bytes",
        ));
    }

    Ok(data)
}

/// Validates an image/label pair: every label within range and exactly
/// `pixels_per_image` image bytes per label. Reports under the
/// `label data` artifact (in-memory parts, not an archive) like before.
pub(crate) fn check_image_label_parts(
    images: &[u8],
    labels: &[u8],
    pixels_per_image: usize,
    max_label: u8,
    operation: &'static str,
) -> Result<()> {
    if labels.iter().any(|&label| label > max_label) {
        return Err(Error::MalformedArtifact {
            operation,
            artifact: "label data",
            reason: ErrorMessage::new(format!("labels must be in the range 0..={max_label}")),
        });
    }
    let expected_images = labels.len().checked_mul(pixels_per_image).ok_or({
        Error::ArithmeticOverflow {
            operation,
            expression: "label count * pixels per image",
        }
    })?;
    if images.len() != expected_images {
        return Err(Error::MalformedArtifact {
            operation,
            artifact: "image archive",
            reason: ErrorMessage::new(format!(
                "images/labels mismatch: {} image bytes for {} labels",
                images.len(),
                labels.len()
            )),
        });
    }
    Ok(())
}

/// Fails a [`DataError`] lookup past the end of a flat image buffer.
pub(crate) fn image_window<'a>(
    images: &'a [u8],
    index: usize,
    pixels_per_image: usize,
    what: &str,
) -> core::result::Result<&'a [u8], DataError> {
    let start = index
        .checked_mul(pixels_per_image)
        .ok_or_else(|| DataError::Dataset(format!("{what} image offset overflow")))?;
    let end = start
        .checked_add(pixels_per_image)
        .ok_or_else(|| DataError::Dataset(format!("{what} image end offset overflow")))?;
    images
        .get(start..end)
        .ok_or_else(|| DataError::Dataset(format!("{what} image buffer is truncated")))
}
