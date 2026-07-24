use crate::err::{Error, Result};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

/// Information summary of an inspected model file.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub format: String,
    pub path: String,
    pub file_size_bytes: u64,
    pub tensor_count: usize,
    pub tensors: Vec<TensorMetaInfo>,
}

#[derive(Debug, Clone)]
pub struct TensorMetaInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub size_bytes: usize,
}

/// Inspects a `.safetensors`, `.gguf`, or `.onnx` file metadata.
pub fn inspect_file<P: AsRef<Path>>(path: P) -> Result<ModelInfo> {
    let path_ref = path.as_ref();
    let file = File::open(path_ref)?;
    let metadata = file.metadata()?;
    let file_size_bytes = metadata.len();
    let path_str = path_ref.to_string_lossy().to_string();

    if path_str.ends_with(".safetensors") {
        inspect_safetensors(file, &path_str, file_size_bytes)
    } else if path_str.ends_with(".gguf") {
        inspect_gguf(file, &path_str, file_size_bytes)
    } else if path_str.ends_with(".onnx") {
        Ok(ModelInfo {
            format: "ONNX Protocol Buffer".to_string(),
            path: path_str,
            file_size_bytes,
            tensor_count: 0,
            tensors: Vec::new(),
        })
    } else {
        Err(Error::Msg(format!(
            "Unsupported model format for inspection: {}",
            path_str
        )))
    }
}

fn inspect_safetensors(mut file: File, path: &str, file_size_bytes: u64) -> Result<ModelInfo> {
    let mut header_size_bytes = [0u8; 8];
    file.read_exact(&mut header_size_bytes)?;
    let header_len = u64::from_le_bytes(header_size_bytes) as usize;

    let mut header_buf = vec![0u8; header_len];
    file.read_exact(&mut header_buf)?;

    let json_val: serde_json::Value = serde_json::from_slice(&header_buf)
        .map_err(|e| Error::Msg(format!("Invalid safetensors header JSON: {}", e)))?;

    let mut tensors = Vec::new();
    if let Some(map) = json_val.as_object() {
        for (name, val) in map {
            if name == "__metadata__" {
                continue;
            }
            let dtype = val
                .get("dtype")
                .and_then(|v| v.as_str())
                .unwrap_or("F32")
                .to_string();
            let shape: Vec<usize> = val
                .get("shape")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|d| d.as_u64().map(|n| n as usize))
                        .collect()
                })
                .unwrap_or_default();

            let elem_count: usize = shape.iter().product();
            let bytes_per_elem = match dtype.as_str() {
                "F64" | "I64" => 8,
                "F32" | "I32" => 4,
                "F16" | "BF16" => 2,
                _ => 1,
            };

            tensors.push(TensorMetaInfo {
                name: name.clone(),
                shape,
                dtype,
                size_bytes: elem_count * bytes_per_elem,
            });
        }
    }

    let tensor_count = tensors.len();
    Ok(ModelInfo {
        format: "SafeTensors Checkpoint".to_string(),
        path: path.to_string(),
        file_size_bytes,
        tensor_count,
        tensors,
    })
}

/// Maps a GGUF/GGML tensor type id to its display name and (for
/// non-block-quantized types) its size in bytes per element.
fn ggml_type_name(type_id: u32) -> &'static str {
    match type_id {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        6 => "Q5_0",
        7 => "Q5_1",
        8 => "Q8_0",
        9 => "Q8_1",
        12 => "Q4_K",
        13 => "Q5_K",
        14 => "Q6_K",
        24 => "I8",
        25 => "I16",
        26 => "I32",
        28 => "BF16",
        other => {
            let _ = other;
            "UNKNOWN"
        }
    }
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_gguf_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64(r)? as usize;
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

/// Skips over a single GGUF metadata value of the given type id, advancing
/// past it without allocating for fixed-size scalars.
fn skip_gguf_value<R: Read + Seek>(r: &mut R, type_id: u32) -> Result<()> {
    match type_id {
        0 | 1 | 7 => {
            r.seek(SeekFrom::Current(1))?;
        }
        2 | 3 => {
            r.seek(SeekFrom::Current(2))?;
        }
        4..=6 => {
            r.seek(SeekFrom::Current(4))?;
        }
        10..=12 => {
            r.seek(SeekFrom::Current(8))?;
        }
        8 => {
            let _ = read_gguf_string(r)?;
        }
        9 => {
            let elem_type = read_u32(r)?;
            let count = read_u64(r)?;
            for _ in 0..count {
                skip_gguf_value(r, elem_type)?;
            }
        }
        other => {
            return Err(Error::Msg(format!(
                "Unknown GGUF metadata value type id: {}",
                other
            )));
        }
    }
    Ok(())
}

fn inspect_gguf(mut file: File, path: &str, file_size_bytes: u64) -> Result<ModelInfo> {
    let mut magic = [0u8; 4];
    file.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        return Err(Error::Msg(
            "Not a valid GGUF file (magic header mismatch)".to_string(),
        ));
    }

    let version = read_u32(&mut file)?;
    let tensor_count = read_u64(&mut file)? as usize;
    let metadata_count = read_u64(&mut file)?;

    // Metadata KV entries are only needed to skip past them to reach the
    // tensor info table; their values aren't surfaced in `ModelInfo` today.
    for _ in 0..metadata_count {
        let _key = read_gguf_string(&mut file)?;
        let type_id = read_u32(&mut file)?;
        skip_gguf_value(&mut file, type_id)?;
    }

    struct RawTensorHeader {
        name: String,
        dims: Vec<u64>,
        ggml_type: u32,
        offset: u64,
    }

    let mut headers = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = read_gguf_string(&mut file)?;
        let n_dims = read_u32(&mut file)?;
        let mut dims = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dims.push(read_u64(&mut file)?);
        }
        let ggml_type = read_u32(&mut file)?;
        let offset = read_u64(&mut file)?;
        headers.push(RawTensorHeader {
            name,
            dims,
            ggml_type,
            offset,
        });
    }

    // Tensor data begins right after the info table, at the file's next
    // absolute position — offsets in the table above are relative to it.
    let data_section_start = file.stream_position()?;

    let mut tensors = Vec::with_capacity(headers.len());
    for (i, h) in headers.iter().enumerate() {
        // Exact byte length is derived from consecutive offsets rather
        // than `dims * bytes_per_elem`, since block-quantized types
        // (Q8_0, Q4_0, ...) don't have a fixed per-element byte size.
        let size_bytes = if i + 1 < headers.len() {
            (headers[i + 1].offset - h.offset) as usize
        } else {
            (file_size_bytes - data_section_start - h.offset) as usize
        };

        // GGUF stores dims reversed (fastest-varying first); reverse back
        // to the natural, row-major shape order for display.
        let shape: Vec<usize> = h.dims.iter().rev().map(|&d| d as usize).collect();

        tensors.push(TensorMetaInfo {
            name: h.name.clone(),
            shape,
            dtype: ggml_type_name(h.ggml_type).to_string(),
            size_bytes,
        });
    }

    Ok(ModelInfo {
        format: format!("GGUF v{}", version),
        path: path.to_string(),
        file_size_bytes,
        tensor_count,
        tensors,
    })
}
