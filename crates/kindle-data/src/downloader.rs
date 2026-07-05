use anyhow::Result;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

pub struct Downloader;

impl Downloader {
    pub fn download(url: &str, cache_dir: &Path, filename: &str) -> Result<PathBuf> {
        let dest_path = cache_dir.join(filename);
        
        if dest_path.exists() {
            return Ok(dest_path);
        }

        std::fs::create_dir_all(cache_dir)?;
        
        let response = ureq::get(url).call().map_err(|e| anyhow::anyhow!("Failed to download {}: {}", url, e))?;
        let mut reader = response.into_body().into_reader();
        
        let mut file = File::create(&dest_path)?;
        io::copy(&mut reader, &mut file)?;
        
        Ok(dest_path)
    }

    pub fn download_and_extract_gz(url: &str, cache_dir: &Path, filename: &str) -> Result<PathBuf> {
        let gz_path = Self::download(url, cache_dir, &format!("{}.gz", filename))?;
        let dest_path = cache_dir.join(filename);
        
        if dest_path.exists() {
            return Ok(dest_path);
        }

        let gz_file = File::open(gz_path)?;
        let mut decoder = flate2::read::GzDecoder::new(gz_file);
        let mut out_file = File::create(&dest_path)?;
        io::copy(&mut decoder, &mut out_file)?;
        
        Ok(dest_path)
    }
}
