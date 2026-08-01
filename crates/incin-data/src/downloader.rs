use anyhow::Result;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

/// Downloader.
pub struct Downloader;

fn validate_relative_filename(filename: &str) -> Result<()> {
    if filename.is_empty()
        || filename.contains("..")
        || filename.contains('/')
        || filename.contains('\\')
        || filename.contains('\0')
        || Path::new(filename).is_absolute()
    {
        return Err(anyhow::anyhow!("Invalid asset filename: '{}'", filename));
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

        std::fs::create_dir_all(cache_dir)?;

        let response = ureq::get(url)
            .call()
            .map_err(|e| anyhow::anyhow!("Failed to download {}: {}", url, e))?;
        let mut reader = response.into_body().into_reader();

        let tmp_filename = format!("{}.tmp.{}", filename, std::process::id());
        let tmp_path = cache_dir.join(&tmp_filename);

        let mut file = File::create(&tmp_path)?;
        let res = io::copy(&mut reader, &mut file);
        if let Err(e) = res {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(e.into());
        }
        file.sync_all()?;
        drop(file);

        std::fs::rename(&tmp_path, &dest_path)?;
        Ok(dest_path)
    }

    /// Download and extract gz.
    pub fn download_and_extract_gz(url: &str, cache_dir: &Path, filename: &str) -> Result<PathBuf> {
        validate_relative_filename(filename)?;
        let gz_filename = format!("{}.gz", filename);
        let gz_path = Self::download(url, cache_dir, &gz_filename)?;
        let dest_path = cache_dir.join(filename);

        if dest_path.exists() {
            return Ok(dest_path);
        }

        let gz_file = File::open(&gz_path)?;
        let mut decoder = flate2::read::GzDecoder::new(gz_file);

        let tmp_filename = format!("{}.tmp.{}", filename, std::process::id());
        let tmp_path = cache_dir.join(&tmp_filename);

        let mut out_file = File::create(&tmp_path)?;
        let res = io::copy(&mut decoder, &mut out_file);
        if let Err(e) = res {
            let _ = std::fs::remove_file(&tmp_path);
            return Err(e.into());
        }
        out_file.sync_all()?;
        drop(out_file);

        std::fs::rename(&tmp_path, &dest_path)?;
        Ok(dest_path)
    }
}
