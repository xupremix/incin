use crate::err::{Error, Result};
use crate::nn::StateDict;
use crate::tensor::backend::Backend;
use crate::tensor::dtype::{FloatDType, Q8_0};
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use core::any::type_name;
use std::fs::File;
use std::io::{BufWriter, Seek, Write};
use std::path::Path;

/// Supported GGUF quantization schemes.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantScheme {
    /// Full precision 32-bit floating point (GGML_TYPE_F32).
    F32,
    /// Half precision 16-bit floating point (GGML_TYPE_F16).
    F16,
    /// 8-bit symmetric quantization with 32-element blocks (GGML_TYPE_Q8_0).
    Q8_0,
    /// 4-bit symmetric quantization with 32-element blocks (GGML_TYPE_Q4_0, W4A16).
    W4A16_Q4_0,
    /// 4-bit K-quant medium quantization (GGML_TYPE_Q4_K, W4A16).
    W4A16_Q4_K_M,
}

impl QuantScheme {
    /// Returns the corresponding GGML type ID for GGUF serialization.
    pub fn ggml_type_id(&self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::W4A16_Q4_0 => 2,
            Self::Q8_0 => 8,
            Self::W4A16_Q4_K_M => 12,
        }
    }

    /// Returns the GGUF file_type metadata integer.
    pub fn file_type_id(&self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::W4A16_Q4_0 => 2,
            Self::Q8_0 => 7,
            Self::W4A16_Q4_K_M => 15,
        }
    }
}

/// Key-Value metadata entry for GGUF headers.
#[derive(Debug, Clone)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    Str(String),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufValue {
    fn type_id(&self) -> u32 {
        match self {
            Self::Uint8(_) => 0,
            Self::Int8(_) => 1,
            Self::Uint16(_) => 2,
            Self::Int16(_) => 3,
            Self::Uint32(_) => 4,
            Self::Int32(_) => 5,
            Self::Float32(_) => 6,
            Self::Bool(_) => 7,
            Self::Str(_) => 8,
            Self::Uint64(_) => 10,
            Self::Int64(_) => 11,
            Self::Float64(_) => 12,
        }
    }

    fn write_binary<W: Write>(&self, w: &mut W) -> Result<()> {
        w.write_all(&self.type_id().to_le_bytes())?;
        match self {
            Self::Uint8(v) => w.write_all(&[*v])?,
            Self::Int8(v) => w.write_all(&[*v as u8])?,
            Self::Uint16(v) => w.write_all(&v.to_le_bytes())?,
            Self::Int16(v) => w.write_all(&v.to_le_bytes())?,
            Self::Uint32(v) => w.write_all(&v.to_le_bytes())?,
            Self::Int32(v) => w.write_all(&v.to_le_bytes())?,
            Self::Float32(v) => w.write_all(&v.to_le_bytes())?,
            Self::Bool(v) => w.write_all(&[*v as u8])?,
            Self::Str(s) => {
                let bytes = s.as_bytes();
                w.write_all(&(bytes.len() as u64).to_le_bytes())?;
                w.write_all(bytes)?;
            }
            Self::Uint64(v) => w.write_all(&v.to_le_bytes())?,
            Self::Int64(v) => w.write_all(&v.to_le_bytes())?,
            Self::Float64(v) => w.write_all(&v.to_le_bytes())?,
        }
        Ok(())
    }
}

/// GGUF metadata container for model architecture properties.
#[derive(Debug, Clone, Default)]
pub struct GgufMetadata {
    pub entries: BTreeMap<String, GgufValue>,
}

impl GgufMetadata {
    pub fn new(arch: &str) -> Self {
        let mut meta = Self::default();
        meta.set("general.architecture", GgufValue::Str(arch.to_string()));
        meta.set(
            "general.producer.name",
            GgufValue::Str("incin-v0.2.0".to_string()),
        );
        meta.set("general.alignment", GgufValue::Uint32(32));
        meta
    }

    pub fn set(&mut self, key: impl Into<String>, val: GgufValue) {
        self.entries.insert(key.into(), val);
    }
}

/// Exporter for saving `incin` modules to GGUF v3 format.
pub struct GgufExporter<'a, B: Backend, M: StateDict<B>> {
    module: &'a M,
    metadata: GgufMetadata,
    quant: QuantScheme,
    _phantom: core::marker::PhantomData<B>,
}

impl<'a, B: Backend, M: StateDict<B>> GgufExporter<'a, B, M> {
    /// Creates a new exporter for the given module, auto-deriving architecture metadata.
    pub fn from_module(module: &'a M) -> Self {
        let full_name = type_name::<M>();
        let short_name = full_name
            .split("::")
            .last()
            .unwrap_or("model")
            .to_lowercase();
        let arch = if short_name.contains("llama") {
            "llama"
        } else if short_name.contains("resnet") {
            "resnet"
        } else {
            "custom"
        };

        let mut metadata = GgufMetadata::new(arch);
        metadata.set("general.name", GgufValue::Str(short_name));

        Self {
            module,
            metadata,
            quant: QuantScheme::F32,
            _phantom: core::marker::PhantomData,
        }
    }

    /// Sets a custom metadata property.
    pub fn with_metadata_entry(mut self, key: impl Into<String>, val: GgufValue) -> Self {
        self.metadata.set(key, val);
        self
    }

    /// Configures the quantization scheme for exported weights.
    pub fn with_quantization(mut self, quant: QuantScheme) -> Self {
        self.quant = quant;
        self
    }

    /// Exports the module and its weights to a `.gguf` file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()>
    where
        B::FloatElem: FloatDType,
    {
        // Only F32 (passthrough) and Q8_0 (real block quantization) are
        // actually backed by a working conversion right now. Refuse the
        // rest rather than silently writing float bytes under a
        // quantized `ggml_type` header, which would produce a `.gguf`
        // file that lies about its own binary layout.
        if !matches!(self.quant, QuantScheme::F32 | QuantScheme::Q8_0) {
            return Err(Error::Msg(format!(
                "GGUF export: quantization scheme {:?} is not yet implemented (only F32 and Q8_0 are supported)",
                self.quant
            )));
        }

        let mut file = BufWriter::new(File::create(path)?);

        // 1. Magic bytes: "GGUF" = 0x46554747
        file.write_all(b"GGUF")?;
        // Version: 3
        file.write_all(&3u32.to_le_bytes())?;

        let mut mapped_tensors = BTreeMap::new();
        self.module.state_dict("", &mut mapped_tensors);
        let tensor_count = mapped_tensors.len() as u64;

        // Auto-set file_type metadata
        let mut final_metadata = self.metadata.clone();
        final_metadata.set(
            "general.file_type",
            GgufValue::Uint32(self.quant.file_type_id()),
        );
        let metadata_count = final_metadata.entries.len() as u64;

        file.write_all(&tensor_count.to_le_bytes())?;
        file.write_all(&metadata_count.to_le_bytes())?;

        // 2. Write KV metadata entries
        for (key, val) in &final_metadata.entries {
            let key_bytes = key.as_bytes();
            file.write_all(&(key_bytes.len() as u64).to_le_bytes())?;
            file.write_all(key_bytes)?;
            val.write_binary(&mut file)?;
        }

        // 3. Collect tensor information table & payloads
        let mut payload_bytes: Vec<u8> = Vec::new();
        let mut tensor_headers = Vec::new();
        let alignment = 32usize;

        for (name, var) in mapped_tensors {
            let shape = B::shape::<B::FloatElem>(var.inner());
            let numel: usize = shape.iter().product();

            // Q8_0 quantizes in blocks of 32 elements; tensors that don't
            // divide evenly (e.g. 1D biases/norm weights) are kept at F32,
            // matching how llama.cpp itself only quantizes eligible weight
            // tensors and leaves the rest at full precision.
            let can_quantize =
                self.quant == QuantScheme::Q8_0 && numel > 0 && numel.is_multiple_of(32);

            let (bytes, ggml_type) = if can_quantize {
                let quantized = B::quantize::<B::FloatElem, Q8_0>(var.inner())?;
                (
                    B::to_bytes::<Q8_0>(&quantized)?,
                    QuantScheme::Q8_0.ggml_type_id(),
                )
            } else {
                (
                    B::to_bytes::<B::FloatElem>(var.inner())?,
                    QuantScheme::F32.ggml_type_id(),
                )
            };
            let n_dims = shape.len() as u32;

            // GGUF stores dimensions in reverse (row-major contiguous first)
            let mut gguf_shape: Vec<u64> = shape.iter().rev().map(|&d| d as u64).collect();
            if gguf_shape.is_empty() {
                gguf_shape.push(1);
            }

            // Pad current payload to 32-byte alignment
            let padding = (alignment - (payload_bytes.len() % alignment)) % alignment;
            payload_bytes.extend(core::iter::repeat_n(0u8, padding));
            let data_offset = payload_bytes.len() as u64;

            payload_bytes.extend(bytes);

            tensor_headers.push((name, n_dims, gguf_shape, ggml_type, data_offset));
        }

        // Write Tensor Information Table
        for (name, n_dims, shape, ggml_type, offset) in tensor_headers {
            let name_bytes = name.as_bytes();
            file.write_all(&(name_bytes.len() as u64).to_le_bytes())?;
            file.write_all(name_bytes)?;
            file.write_all(&n_dims.to_le_bytes())?;
            for dim in shape {
                file.write_all(&dim.to_le_bytes())?;
            }
            file.write_all(&ggml_type.to_le_bytes())?;
            file.write_all(&offset.to_le_bytes())?;
        }

        // Write 32-byte alignment padding before binary payload
        let current_pos = file.stream_position()?;
        let header_padding = (alignment - (current_pos as usize % alignment)) % alignment;
        file.write_all(&vec![0u8; header_padding])?;

        // 4. Write Tensor Binary Payload
        file.write_all(&payload_bytes)?;
        file.flush()?;

        Ok(())
    }
}
