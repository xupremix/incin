use crate::dist::placement::Local;
use crate::err::{Error, Result};
use crate::exec::catalog::{QuantizationAttributes, op};
use crate::exec::request::TensorHandle;
use crate::exec::{self, ExecutionContext};
use crate::nn::{StateSnapshot, VisitState};
use crate::tensor::backend::{Backend, Execute, HostInterop, SupportsDType};
use crate::tensor::dtype::{DTypeId, Q8_0};
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

fn encode_q4_0(bytes: &[u8]) -> Option<Vec<u8>> {
    let mut encoded = Vec::with_capacity(bytes.len() / 128 * 18);
    let mut values = [0.0f32; 32];
    for block in bytes.chunks_exact(128) {
        for (index, value) in values.iter_mut().enumerate() {
            let offset = index * 4;
            *value = f32::from_ne_bytes([
                block[offset],
                block[offset + 1],
                block[offset + 2],
                block[offset + 3],
            ]);
            if !value.is_finite() {
                return None;
            }
        }
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &value in &values {
            let abs = value.abs();
            if abs > amax {
                amax = abs;
                max = value;
            }
        }
        let scale = max / -8.0;
        let scale_half = half::f16::from_f32(scale);
        if !scale_half.is_finite() || (amax != 0.0 && scale_half == half::f16::ZERO) {
            return None;
        }
        let inverse_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };
        if !inverse_scale.is_finite() {
            return None;
        }
        encoded.extend_from_slice(&scale_half.to_le_bytes());
        let quantize = |value: f32| (value * inverse_scale + 8.5).clamp(0.0, 15.0) as u8;
        for index in 0..16 {
            encoded.push(quantize(values[index]) | (quantize(values[index + 16]) << 4));
        }
    }
    Some(encoded)
}

/// Rounds `fval` to the nearest integer exactly like llama.cpp's
/// `nearest_int` in `ggml-quants.c`: the `+12582912.0` (1.5 * 2^23) add
/// rounds the fraction away through float addition (ties-to-even), and the
/// mantissa bits then encode the integer result.
fn ggml_nearest_int(fval: f32) -> i32 {
    let val = fval + 12582912.0;
    ((val.to_bits() & 0x007f_ffff) as i32) - 0x0040_0000
}

/// Reads the packed 6-bit scale/min pair `j` out of a Q4_K superblock's
/// 12-byte `scales` array; mirrors `get_scale_min_k4` in llama.cpp
/// `ggml-quants.c` (and identically `dequantize_row_q4_K`'s use of it).
fn get_scale_min_k4(j: usize, scales: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0f) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

/// First stage of Q4_K quantization: fits one 32-element sub-block to the
/// `scale * q - min` grid by weighted least squares, mirroring
/// `make_qkx2_quants(n, nmax=15, ..., rmin=-1, rdelta=0.1, nstep=20,
/// use_mad=false)` from llama.cpp `ggml-quants.c` as called by
/// `quantize_row_q4_K_ref`. Returns `(scale, min)` with `min >= 0` so the
/// dequantized value is `scale * q - min`.
fn make_qkx2_quants(
    x: &[f32],
    weights: &[f32],
    l: &mut [u8],
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
) -> (f32, f32) {
    const NMAX: i32 = 15;
    let n = x.len();
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        if x[i] < min {
            min = x[i];
        }
        if x[i] > max {
            max = x[i];
        }
        sum_w += weights[i];
        sum_x += weights[i] * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        l.fill(0);
        return (0.0, -min);
    }
    let iscale = NMAX as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_error = 0.0f32;
    for i in 0..n {
        let li = ggml_nearest_int(iscale * (x[i] - min)).clamp(0, NMAX) as u8;
        l[i] = li;
        let diff = scale * f32::from(li) + min - x[i];
        best_error += weights[i] * (diff * diff);
    }
    if nstep >= 1 {
        for step in 0..=nstep {
            let iscale = (rmin + rdelta * step as f32 + NMAX as f32) / (max - min);
            let mut sum_l = 0.0f32;
            let mut sum_l2 = 0.0f32;
            let mut sum_xl = 0.0f32;
            for i in 0..n {
                let li = ggml_nearest_int(iscale * (x[i] - min));
                laux[i] = li as u8;
                let lf = f32::from(laux[i]);
                sum_l += weights[i] * lf;
                sum_l2 += weights[i] * lf * lf;
                sum_xl += weights[i] * lf * x[i];
            }
            let denom = sum_w * sum_l2 - sum_l * sum_l;
            if denom > 0.0 {
                let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / denom;
                let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / denom;
                if this_min > 0.0 {
                    this_min = 0.0;
                    this_scale = sum_xl / sum_l2;
                }
                let mut cur_error = 0.0f32;
                for i in 0..n {
                    let diff = this_scale * f32::from(laux[i]) + this_min - x[i];
                    cur_error += weights[i] * (diff * diff);
                }
                if cur_error < best_error {
                    l.copy_from_slice(laux);
                    best_error = cur_error;
                    scale = this_scale;
                    min = this_min;
                }
            }
        }
    }
    (scale, -min)
}

/// Encodes native-endian F32 bytes as GGML Q4_K superblocks: 256 elements
/// per 144-byte block (fp16 `d`/`dmin`, 8 packed 6-bit scales + 8 packed
/// 6-bit mins, 128 bytes of 4-bit quants), following
/// `quantize_row_q4_K_ref` in llama.cpp
/// <https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-quants.c>.
/// Returns `None` when the data has no faithful Q4_K representation
/// (non-finite values, overflowed intermediates, or fp16 scale/min that
/// would round to zero or infinity while carrying signal); callers keep the
/// whole tensor in F32 rather than mixing encodings.
fn encode_q4_k(bytes: &[u8]) -> Option<Vec<u8>> {
    const QK_K: usize = 256;
    const SUB_BLOCK: usize = 32;
    let mut encoded = Vec::with_capacity(bytes.len() / (QK_K * 4) * 144);
    let mut values = [0.0f32; QK_K];
    let mut weights = [0.0f32; SUB_BLOCK];
    let mut labels = [0u8; QK_K];
    let mut laux = [0u8; SUB_BLOCK];
    let mut scales = [0.0f32; QK_K / SUB_BLOCK];
    let mut mins = [0.0f32; QK_K / SUB_BLOCK];
    for block in bytes.chunks_exact(QK_K * 4) {
        for (index, value) in values.iter_mut().enumerate() {
            let offset = index * 4;
            *value = f32::from_ne_bytes([
                block[offset],
                block[offset + 1],
                block[offset + 2],
                block[offset + 3],
            ]);
            if !value.is_finite() {
                return None;
            }
        }
        let mut max_scale = 0.0f32;
        let mut max_min = 0.0f32;
        for j in 0..QK_K / SUB_BLOCK {
            let sub = &values[j * SUB_BLOCK..(j + 1) * SUB_BLOCK];
            let sum_x2: f32 = sub.iter().map(|value| value * value).sum();
            let av_x = (sum_x2 / SUB_BLOCK as f32).sqrt();
            if !av_x.is_finite() {
                return None;
            }
            for (weight, value) in weights.iter_mut().zip(sub) {
                *weight = av_x + value.abs();
            }
            let (scale, min) = make_qkx2_quants(
                sub,
                &weights,
                &mut labels[j * SUB_BLOCK..(j + 1) * SUB_BLOCK],
                &mut laux,
                -1.0,
                0.1,
                20,
            );
            if !scale.is_finite() || !min.is_finite() {
                return None;
            }
            scales[j] = scale;
            mins[j] = min;
            if scale > max_scale {
                max_scale = scale;
            }
            if min > max_min {
                max_min = min;
            }
        }
        let inv_scale = if max_scale > 0.0 {
            63.0 / max_scale
        } else {
            0.0
        };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
        if !inv_scale.is_finite() || !inv_min.is_finite() {
            return None;
        }
        let mut scale_bytes = [0u8; 12];
        for j in 0..QK_K / SUB_BLOCK {
            let ls = (ggml_nearest_int(inv_scale * scales[j]) as u8).min(63);
            let lm = (ggml_nearest_int(inv_min * mins[j]) as u8).min(63);
            if j < 4 {
                scale_bytes[j] = ls;
                scale_bytes[j + 4] = lm;
            } else {
                scale_bytes[j + 4] = (ls & 0x0f) | ((lm & 0x0f) << 4);
                scale_bytes[j - 4] |= (ls >> 4) << 6;
                scale_bytes[j] |= (lm >> 4) << 6;
            }
        }
        let d = half::f16::from_f32(max_scale / 63.0);
        let dmin = half::f16::from_f32(max_min / 63.0);
        if !d.is_finite() || !dmin.is_finite() {
            return None;
        }
        if (max_scale != 0.0 && d == half::f16::ZERO) || (max_min != 0.0 && dmin == half::f16::ZERO)
        {
            return None;
        }
        if d == half::f16::ZERO
            && dmin == half::f16::ZERO
            && values.iter().any(|value| *value != 0.0)
        {
            return None;
        }
        let d_f = d.to_f32();
        let dmin_f = dmin.to_f32();
        for j in 0..QK_K / SUB_BLOCK {
            let (sc, m) = get_scale_min_k4(j, &scale_bytes);
            let sub_scale = d_f * f32::from(sc);
            if sub_scale == 0.0 {
                continue;
            }
            let sub_min = dmin_f * f32::from(m);
            for ii in 0..SUB_BLOCK {
                let index = j * SUB_BLOCK + ii;
                let ratio = (values[index] + sub_min) / sub_scale;
                if !ratio.is_finite() {
                    return None;
                }
                labels[index] = ggml_nearest_int(ratio).clamp(0, 15) as u8;
            }
        }
        let mut qs = [0u8; QK_K / 2];
        for j in (0..QK_K).step_by(64) {
            for index in 0..32 {
                qs[j / 2 + index] = labels[j + index] | (labels[j + index + 32] << 4);
            }
        }
        encoded.extend_from_slice(&d.to_le_bytes());
        encoded.extend_from_slice(&dmin.to_le_bytes());
        encoded.extend_from_slice(&scale_bytes);
        encoded.extend_from_slice(&qs);
    }
    Some(encoded)
}

fn f32_bytes_to_little_endian(bytes: &[u8]) -> Vec<u8> {
    bytes
        .chunks_exact(4)
        .flat_map(|chunk| {
            f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]).to_le_bytes()
        })
        .collect()
}

/// Key-Value metadata entry for GGUF headers.
#[derive(Debug, Clone)]
pub enum GgufValue {
    /// GGUF `Uint8` metadata value.
    Uint8(u8),
    /// GGUF `Int8` metadata value.
    Int8(i8),
    /// GGUF `Uint16` metadata value.
    Uint16(u16),
    /// GGUF `Int16` metadata value.
    Int16(i16),
    /// GGUF `Uint32` metadata value.
    Uint32(u32),
    /// GGUF `Int32` metadata value.
    Int32(i32),
    /// GGUF `Float32` metadata value.
    Float32(f32),
    /// GGUF `Bool` metadata value.
    Bool(bool),
    /// GGUF string metadata value.
    Str(String),
    /// GGUF `Uint64` metadata value.
    Uint64(u64),
    /// GGUF `Int64` metadata value.
    Int64(i64),
    /// GGUF `Float64` metadata value.
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
                let len = u64::try_from(bytes.len())
                    .map_err(|_| Error::Msg("GGUF string is too large for the format".into()))?;
                w.write_all(&len.to_le_bytes())?;
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
    /// Metadata entries keyed by name.
    pub entries: BTreeMap<String, GgufValue>,
}

impl GgufMetadata {
    /// Creates metadata for one architecture.
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

    /// Sets one metadata entry.
    pub fn set(&mut self, key: impl Into<String>, val: GgufValue) {
        self.entries.insert(key.into(), val);
    }
}

/// Exporter for saving `incin` modules to GGUF v3 format.
pub struct GgufExporter<
    'a,
    B: Backend + crate::tensor::backend::VariableBackend + HostInterop,
    M: VisitState<B>,
> {
    module: &'a M,
    metadata: GgufMetadata,
    quant: QuantScheme,
    _phantom: core::marker::PhantomData<B>,
}

impl<'a, B, M> GgufExporter<'a, B, M>
where
    B: Backend + crate::tensor::backend::VariableBackend + Execute<op::Quantize> + HostInterop,
    <B as Execute<op::Quantize>>::Output: Into<B::Storage<Q8_0>>,
    M: VisitState<B>,
{
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
        B: SupportsDType<f32>,
    {
        // Only F32 (passthrough) and the working block quantizations
        // (Q8_0, Q4_0, Q4_K) are backed by working conversions. Refuse the
        // rest rather than silently writing float bytes under a quantized
        // `ggml_type` header, which would produce a `.gguf` file that lies
        // about its own binary layout.
        if !matches!(
            self.quant,
            QuantScheme::F32
                | QuantScheme::Q8_0
                | QuantScheme::W4A16_Q4_0
                | QuantScheme::W4A16_Q4_K_M
        ) {
            return Err(Error::Msg(format!(
                "GGUF export: quantization scheme {:?} is not yet implemented (only F32, Q8_0, Q4_0 and Q4_K are supported)",
                self.quant
            )));
        }

        let snapshot: StateSnapshot = crate::nn::collect_state::<B, _>(self.module)?;
        for (_, value) in snapshot.iter() {
            if value.shape().dims().len() > 4 {
                return Err(Error::InvalidModuleState {
                    operation: "GGUF export",
                    reason: crate::err::ErrorMessage::new(
                        "GGUF supports at most 4 tensor dimensions",
                    ),
                });
            }
        }

        let mut file = BufWriter::new(File::create(path)?);

        // 1. Magic bytes: "GGUF" = 0x46554747
        file.write_all(b"GGUF")?;
        // Version: 3
        file.write_all(&3u32.to_le_bytes())?;

        let tensor_count = u64::try_from(snapshot.len())
            .map_err(|_| Error::Msg("tensor count is too large for the GGUF format".into()))?;

        // Auto-set file_type metadata
        let mut final_metadata = self.metadata.clone();
        final_metadata.set(
            "general.file_type",
            GgufValue::Uint32(self.quant.file_type_id()),
        );
        let metadata_count = u64::try_from(final_metadata.entries.len())
            .map_err(|_| Error::Msg("metadata count is too large for the GGUF format".into()))?;

        file.write_all(&tensor_count.to_le_bytes())?;
        file.write_all(&metadata_count.to_le_bytes())?;

        // 2. Write KV metadata entries
        for (key, val) in &final_metadata.entries {
            let key_bytes = key.as_bytes();
            let key_len = u64::try_from(key_bytes.len())
                .map_err(|_| Error::Msg("GGUF metadata key is too large".into()))?;
            file.write_all(&key_len.to_le_bytes())?;
            file.write_all(key_bytes)?;
            val.write_binary(&mut file)?;
        }

        // 3. Collect tensor information table & payloads
        let mut payload_bytes: Vec<u8> = Vec::new();
        let mut tensor_headers = Vec::new();
        let alignment = 32usize;

        for (name, value) in snapshot.iter() {
            let shape = value.shape().dims();
            if value.dtype().builtin_id() != Some(DTypeId::F32) {
                return Err(Error::Msg(format!(
                    "GGUF export currently requires F32 state, got {} for {}",
                    value.dtype().name(),
                    name
                )));
            }
            let numel = crate::shapes::ShapeBuf::from_slice(shape)
                .checked_numel(crate::shapes::error::OperationKind::Storage)?;

            // GGUF identifies a tensor's rows by its last (fastest-varying)
            // dimension, so block quantization needs a positive element
            // count and a last dimension that is a multiple of the scheme's
            // row size (32, or 256 for Q4_K superblocks); tensors that do
            // not qualify stay F32.
            let row_multiple = match self.quant {
                QuantScheme::W4A16_Q4_K_M => 256,
                _ => 32,
            };
            let can_quantize = matches!(
                self.quant,
                QuantScheme::Q8_0 | QuantScheme::W4A16_Q4_0 | QuantScheme::W4A16_Q4_K_M
            ) && numel > 0
                && shape
                    .last()
                    .is_some_and(|last| last.is_multiple_of(row_multiple));

            let (bytes, ggml_type) = if can_quantize && self.quant == QuantScheme::W4A16_Q4_K_M {
                match encode_q4_k(value.bytes()) {
                    Some(encoded) => (encoded, QuantScheme::W4A16_Q4_K_M.ggml_type_id()),
                    None => (
                        f32_bytes_to_little_endian(value.bytes()),
                        QuantScheme::F32.ggml_type_id(),
                    ),
                }
            } else if can_quantize && self.quant == QuantScheme::W4A16_Q4_0 {
                match encode_q4_0(value.bytes()) {
                    Some(encoded) => (encoded, QuantScheme::W4A16_Q4_0.ggml_type_id()),
                    None => (
                        f32_bytes_to_little_endian(value.bytes()),
                        QuantScheme::F32.ggml_type_id(),
                    ),
                }
            } else if can_quantize {
                let storage = B::from_bytes::<f32>(
                    value.bytes(),
                    shape,
                    DTypeId::F32.descriptor(),
                    &crate::tensor::device::DeviceId::cpu(),
                )?;
                let input = TensorHandle::from_storage::<B, f32, Local>(&storage);
                let context = ExecutionContext::from_scope(B::default());
                let quantized = exec::dispatch::execute::<op::Quantize, B>(
                    &context,
                    QuantizationAttributes {
                        dtype: DTypeId::Q8_0.descriptor(),
                    },
                    &[input],
                )?
                .into();
                (
                    B::to_bytes::<Q8_0>(&quantized)?,
                    QuantScheme::Q8_0.ggml_type_id(),
                )
            } else {
                (
                    f32_bytes_to_little_endian(value.bytes()),
                    QuantScheme::F32.ggml_type_id(),
                )
            };

            // GGUF stores dimensions in reverse (row-major contiguous first)
            let mut gguf_shape: Vec<u64> = shape
                .iter()
                .rev()
                .map(|&dimension| {
                    u64::try_from(dimension)
                        .map_err(|_| Error::Msg("tensor dimension is too large for GGUF".into()))
                })
                .collect::<Result<_>>()?;
            if gguf_shape.is_empty() {
                gguf_shape.push(1);
            }
            let n_dims = u32::try_from(gguf_shape.len())
                .map_err(|_| Error::Msg("tensor rank is too large for GGUF".into()))?;

            // Pad current payload to 32-byte alignment
            let padding = (alignment - (payload_bytes.len() % alignment)) % alignment;
            payload_bytes.extend(core::iter::repeat_n(0u8, padding));
            let data_offset = u64::try_from(payload_bytes.len())
                .map_err(|_| Error::Msg("GGUF payload offset exceeds the format".into()))?;

            payload_bytes.extend(bytes);

            tensor_headers.push((name, n_dims, gguf_shape, ggml_type, data_offset));
        }

        // Write Tensor Information Table
        for (name, n_dims, shape, ggml_type, offset) in tensor_headers {
            let name_bytes = name.as_str().as_bytes();
            let name_len = u64::try_from(name_bytes.len())
                .map_err(|_| Error::Msg("GGUF tensor name is too large".into()))?;
            file.write_all(&name_len.to_le_bytes())?;
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
        let current_pos = usize::try_from(current_pos)
            .map_err(|_| Error::Msg("GGUF header position does not fit this platform".into()))?;
        let header_padding = (alignment - (current_pos % alignment)) % alignment;
        file.write_all(&vec![0u8; header_padding])?;

        let payload_padding = (alignment - (payload_bytes.len() % alignment)) % alignment;
        payload_bytes.extend(core::iter::repeat_n(0u8, payload_padding));

        // 4. Write Tensor Binary Payload
        file.write_all(&payload_bytes)?;
        file.flush()?;

        Ok(())
    }
}
