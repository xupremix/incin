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

    /// Downloads a `.tar.gz` archive and extracts the top-level directory
    /// it contains into `cache_dir`, returning that directory.
    ///
    /// `top_dir` names the single directory the archive must produce.
    /// Every member must stay inside it: absolute paths and `..` escapes
    /// are malformed archives, not skipped entries. Extraction lands in a
    /// process-suffixed staging directory renamed into place, so an
    /// interrupted run never leaves a half-extracted tree behind.
    pub fn download_and_extract_tar_gz(
        url: &str,
        cache_dir: &Path,
        tarball_filename: &str,
        top_dir: &str,
    ) -> Result<PathBuf> {
        const OPERATION: &str = "download and extract tarball";
        validate_relative_filename(tarball_filename)?;
        if top_dir.is_empty()
            || top_dir.contains("..")
            || top_dir.contains('/')
            || top_dir.contains('\\')
        {
            return Err(Error::MalformedArtifact {
                operation: OPERATION,
                artifact: "top-level directory",
                reason: ErrorMessage::new("archive top directory must be a single component"),
            });
        }
        let dest_dir = cache_dir.join(top_dir);
        if dest_dir.is_dir() {
            return Ok(dest_dir);
        }
        let gz_path = Self::download(url, cache_dir, tarball_filename)?;
        Self::extract_tar_gz_file(&gz_path, cache_dir, top_dir)
    }

    /// Extracts an already-downloaded `.tar.gz` file, returning the
    /// extracted top directory. Split out so tests cover traversal
    /// refusal and layout checks without a network round trip.
    fn extract_tar_gz_file(gz_path: &Path, cache_dir: &Path, top_dir: &str) -> Result<PathBuf> {
        const OPERATION: &str = "download and extract tarball";
        let dest_dir = cache_dir.join(top_dir);
        let gz_file = File::open(gz_path).map_err(|e| io_error(OPERATION, e))?;
        let decoder = flate2::read::GzDecoder::new(gz_file);
        let mut archive = tar::Archive::new(decoder);
        let stage_dir = cache_dir.join(format!(".tmp-extract-{}", std::process::id()));
        if stage_dir.exists() {
            std::fs::remove_dir_all(&stage_dir).map_err(|e| io_error(OPERATION, e))?;
        }
        std::fs::create_dir_all(&stage_dir).map_err(|e| io_error(OPERATION, e))?;
        Self::unpack_checked(&mut archive, &stage_dir, OPERATION)?;
        let staged_top = stage_dir.join(top_dir);
        if !staged_top.is_dir() {
            let _ = std::fs::remove_dir_all(&stage_dir);
            return Err(Error::MalformedArtifact {
                operation: OPERATION,
                artifact: "tarball layout",
                reason: ErrorMessage::new("archive did not produce the expected top directory"),
            });
        }
        std::fs::rename(&staged_top, &dest_dir).map_err(|e| io_error(OPERATION, e))?;
        let _ = std::fs::remove_dir_all(&stage_dir);
        Ok(dest_dir)
    }

    /// Unpacks every member of `archive` into `stage_dir`, refusing
    /// absolute paths and parent escapes as malformed archives.
    fn unpack_checked<R: std::io::Read>(
        archive: &mut tar::Archive<R>,
        stage_dir: &Path,
        operation: &'static str,
    ) -> Result<()> {
        for member in archive.entries().map_err(|e| io_error(operation, e))? {
            let mut member = member.map_err(|e| io_error(operation, e))?;
            let member_path = member
                .path()
                .map_err(|e| io_error(operation, e))?
                .into_owned();
            if member_path.is_absolute()
                || member_path.components().any(|c| {
                    matches!(
                        c,
                        std::path::Component::ParentDir | std::path::Component::RootDir
                    )
                })
            {
                return Err(Error::MalformedArtifact {
                    operation,
                    artifact: "tarball member",
                    reason: ErrorMessage::new("archive member escapes its directory"),
                });
            }
            member
                .unpack_in(stage_dir)
                .map_err(|e| io_error(operation, e))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(prefix: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "incin-tar-{prefix}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("clock is after the epoch")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("temp dir should create");
        dir
    }

    /// Builds a `.tar.gz` file in `dir` from `(path, bytes)` members.
    fn make_tarball(dir: &Path, name: &str, members: &[(&str, &[u8])]) -> PathBuf {
        let path = dir.join(name);
        let file = File::create(&path).expect("tarball should create");
        let encoder = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        let mut builder = tar::Builder::new(encoder);
        for (member, bytes) in members {
            let mut header = tar::Header::new_gnu();
            header.set_size(bytes.len() as u64);
            header.set_mode(0o644);
            header.set_cksum();
            builder
                .append_data(&mut header, member, *bytes)
                .expect("member should append");
        }
        builder.into_inner().expect("encoder should finish");
        path
    }

    /// Hand-builds a one-member `.tar.gz` whose member name is taken
    /// verbatim: the `tar` builder refuses `..` at append time, so only
    /// raw bytes exercise the extractor's own traversal guard.
    fn make_raw_tarball(dir: &Path, name: &str, member: &str, body: &[u8]) -> PathBuf {
        let path = dir.join(name);
        let mut header = [0u8; 512];
        let name_bytes = member.as_bytes();
        header[..name_bytes.len()].copy_from_slice(name_bytes);
        for (slot, octal) in [
            (100usize, format!("{0:07o}\0", 0o644)),
            (124, format!("{0:011o}\0", body.len())),
        ] {
            let bytes = octal.as_bytes();
            header[slot..slot + bytes.len()].copy_from_slice(bytes);
        }
        header[156] = b'0';
        header[257..262].copy_from_slice(b"ustar");
        // Checksum over spaces in its own field.
        header[148..156].copy_from_slice(b"        ");
        let sum: u32 = header.iter().map(|&b| b as u32).sum();
        let cksum = format!("{sum:06o}\0 ");
        header[148..156].copy_from_slice(cksum.as_bytes());
        let file = File::create(&path).expect("tarball should create");
        let mut encoder = flate2::write::GzEncoder::new(file, flate2::Compression::fast());
        use std::io::Write;
        encoder.write_all(&header).expect("header should write");
        encoder.write_all(body).expect("body should write");
        let pad = (512 - body.len() % 512) % 512;
        encoder
            .write_all(&vec![0u8; pad + 1024])
            .expect("padding should write");
        encoder.finish().expect("encoder should finish");
        path
    }

    #[test]
    fn tarball_extracts_top_dir_and_refuses_escapes() {
        let dir = temp_dir("extract");
        let gz = make_tarball(
            &dir,
            "data.tar.gz",
            &[("top/a.bin", &[1u8, 2, 3][..]), ("top/b.bin", &[4u8][..])],
        );
        let out =
            Downloader::extract_tar_gz_file(&gz, &dir, "top").expect("clean tree should extract");
        assert_eq!(out, dir.join("top"));
        assert_eq!(
            std::fs::read(out.join("a.bin")).expect("member should land"),
            vec![1u8, 2, 3]
        );

        let evil = temp_dir("evil");
        let evil_gz = make_raw_tarball(&evil, "evil.tar.gz", "../escape.bin", &[9u8]);
        let error = Downloader::extract_tar_gz_file(&evil_gz, &evil, "top")
            .expect_err("a parent escape must be refused");
        assert!(
            matches!(
                error,
                Error::MalformedArtifact {
                    artifact: "tarball member",
                    ..
                }
            ),
            "unexpected error: {error:?}"
        );

        let wrong = temp_dir("wrong");
        let wrong_gz = make_tarball(&wrong, "w.tar.gz", &[("other/a.bin", &[1u8][..])]);
        let error = Downloader::extract_tar_gz_file(&wrong_gz, &wrong, "top")
            .expect_err("a missing top dir must be refused");
        assert!(
            matches!(
                error,
                Error::MalformedArtifact {
                    artifact: "tarball layout",
                    ..
                }
            ),
            "unexpected error: {error:?}"
        );
        let _ = std::fs::remove_dir_all(&dir);
        let _ = std::fs::remove_dir_all(&evil);
        let _ = std::fs::remove_dir_all(&wrong);
    }

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
