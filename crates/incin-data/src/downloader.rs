use incin_core::error::{Error, ErrorMessage, Result};
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

/// Downloader.
pub struct Downloader;

/// Wraps an I/O failure with the operation that produced it.
fn io_error(operation: &'static str, source: io::Error) -> Error {
    Error::Io {
        operation,
        message: ErrorMessage::new(source.to_string()),
    }
}

fn validate_relative_filename(filename: &str) -> Result<()> {
    if filename.is_empty()
        || filename.contains("..")
        || filename.contains('/')
        || filename.contains('\\')
        || filename.contains('\0')
        || Path::new(filename).is_absolute()
    {
        return Err(Error::MalformedArtifact {
            operation: "download",
            artifact: "asset filename",
            reason: ErrorMessage::new(format!(
                "asset filename must be a relative single path component, got {filename:?}"
            )),
        });
    }
    Ok(())
}

impl Downloader {
    /// Download.
    pub fn download(url: &str, cache_dir: &Path, filename: &str) -> Result<PathBuf> {
        validate_relative_filename(filename)?;
        let dest_path = cache_dir.join(filename);

        if dest_path.exists() {
            return Ok(dest_path);
        }

        std::fs::create_dir_all(cache_dir).map_err(|e| io_error("download", e))?;

        let response = ureq::get(url).call().map_err(|e| Error::Io {
            operation: "download",
            message: ErrorMessage::new(format!("failed to fetch {url}: {e}")),
        })?;
        let mut reader = response.into_body().into_reader();

        let tmp_filename = format!("{}.tmp.{}", filename, std::process::id());
        let tmp_path = cache_dir.join(&tmp_filename);

        let mut file = File::create(&tmp_path).map_err(|e| io_error("download", e))?;
        let res = io::copy(&mut reader, &mut file);
        if let Err(e) = res {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(io_error("download", e));
        }
        file.sync_all().map_err(|e| io_error("download", e))?;
        drop(file);

        std::fs::rename(&tmp_path, &dest_path).map_err(|e| io_error("download", e))?;
        Ok(dest_path)
    }

    /// Download and extract gz.
    pub fn download_and_extract_gz(url: &str, cache_dir: &Path, filename: &str) -> Result<PathBuf> {
        validate_relative_filename(filename)?;
        const OPERATION: &str = "download and extract";
        let gz_filename = format!("{filename}.gz");
        let gz_path = Self::download(url, cache_dir, &gz_filename)?;
        let dest_path = cache_dir.join(filename);

        if dest_path.exists() {
            return Ok(dest_path);
        }

        let gz_file = File::open(&gz_path).map_err(|e| io_error(OPERATION, e))?;
        let mut decoder = flate2::read::GzDecoder::new(gz_file);

        let tmp_filename = format!("{}.tmp.{}", filename, std::process::id());
        let tmp_path = cache_dir.join(&tmp_filename);

        let mut out_file = File::create(&tmp_path).map_err(|e| io_error(OPERATION, e))?;
        let res = io::copy(&mut decoder, &mut out_file);
        if let Err(e) = res {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(io_error(OPERATION, e));
        }
        out_file.sync_all().map_err(|e| io_error(OPERATION, e))?;
        drop(out_file);

        std::fs::rename(&tmp_path, &dest_path).map_err(|e| io_error(OPERATION, e))?;
        Ok(dest_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn download_rejects_path_escaping_filenames_as_malformed_artifact() {
        for filename in ["../escape", "sub/dir", "abs/C:\\temp", "", "nested\\name"] {
            let error =
                Downloader::download("https://example.invalid/x", Path::new("/tmp"), filename)
                    .expect_err("a path-escaping name must be refused before any request");
            assert!(
                matches!(
                    error,
                    Error::MalformedArtifact {
                        operation: "download",
                        artifact: "asset filename",
                        ..
                    }
                ),
                "unexpected error for {filename:?}"
            );
        }
    }
}
