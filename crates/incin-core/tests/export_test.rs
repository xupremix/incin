//! Integration coverage for `test_gguf_export_and_inspect_roundtrip` on the documented public surface.
extern crate incin_core as incin;
use incin_backends::cpu::CpuBackendImpl;
use incin_core::io::{GgufExporter, MlxExporter, QuantScheme, inspect_file};
use incin_core::nn::{
    StateSnapshot, StateValue, VisitState, VisitStateMut, collect_state, load_state,
};
use incin_core::prelude::*;
use std::collections::BTreeMap;
use std::io::{Cursor, Read};
use tempfile::tempdir;

struct GgufTensorHeader {
    name: String,
    dimensions: Vec<u64>,
    dtype: u32,
    offset: usize,
}

fn read_array<const N: usize>(reader: &mut Cursor<&[u8]>) -> [u8; N] {
    let mut bytes = [0; N];
    reader.read_exact(&mut bytes).unwrap();
    bytes
}

fn read_string(reader: &mut Cursor<&[u8]>) -> String {
    let len = u64::from_le_bytes(read_array(reader)) as usize;
    let mut bytes = vec![0; len];
    reader.read_exact(&mut bytes).unwrap();
    String::from_utf8(bytes).unwrap()
}

fn read_gguf(
    bytes: &[u8],
) -> (
    BTreeMap<String, serde_json::Value>,
    Vec<GgufTensorHeader>,
    usize,
) {
    let mut reader = Cursor::new(bytes);
    assert_eq!(&read_array::<4>(&mut reader), b"GGUF");
    assert_eq!(u32::from_le_bytes(read_array(&mut reader)), 3);
    let tensor_count = u64::from_le_bytes(read_array(&mut reader));
    let metadata_count = u64::from_le_bytes(read_array(&mut reader));
    let mut metadata = BTreeMap::new();
    for _ in 0..metadata_count {
        let key = read_string(&mut reader);
        let value = match u32::from_le_bytes(read_array(&mut reader)) {
            4 => serde_json::json!(u32::from_le_bytes(read_array(&mut reader))),
            8 => serde_json::json!(read_string(&mut reader)),
            dtype => panic!("unexpected fixture metadata type {dtype}"),
        };
        assert!(metadata.insert(key, value).is_none());
    }
    let mut headers = Vec::new();
    for _ in 0..tensor_count {
        let name = read_string(&mut reader);
        let rank = u32::from_le_bytes(read_array(&mut reader));
        let dimensions = (0..rank)
            .map(|_| u64::from_le_bytes(read_array(&mut reader)))
            .collect();
        let dtype = u32::from_le_bytes(read_array(&mut reader));
        let offset = u64::from_le_bytes(read_array(&mut reader)) as usize;
        assert_eq!(offset % 32, 0);
        headers.push(GgufTensorHeader {
            name,
            dimensions,
            dtype,
            offset,
        });
    }
    let header_end = reader.position() as usize;
    let data_start = header_end.next_multiple_of(32);
    assert!(bytes[header_end..data_start].iter().all(|&byte| byte == 0));
    (metadata, headers, data_start)
}

fn set_export_weights<M>(module: &mut M, weights: &[f32])
where
    M: VisitState<CpuBackendImpl> + VisitStateMut<CpuBackendImpl>,
{
    let current = collect_state::<CpuBackendImpl, _>(module).unwrap();
    let mut replacement = StateSnapshot::new();
    for (path, value) in current.iter() {
        let bytes = if path.as_str().is_empty() || path.as_str().ends_with("weight") {
            weights
                .iter()
                .flat_map(|value| value.to_ne_bytes())
                .collect()
        } else {
            value.bytes().to_vec()
        };
        replacement
            .insert(
                path.clone(),
                StateValue::new(value.shape().clone(), value.dtype(), bytes, value.role()).unwrap(),
            )
            .unwrap();
    }
    load_state::<CpuBackendImpl, _>(module, &replacement).unwrap();
}

fn decode_q4_0(block: &[u8]) -> [f32; 32] {
    assert_eq!(block.len(), 18);
    let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    std::array::from_fn(|index| {
        let packed = block[2 + index % 16];
        let nibble = if index < 16 {
            packed & 0x0f
        } else {
            packed >> 4
        };
        (i32::from(nibble) - 8) as f32 * scale
    })
}

/// Reads the packed 6-bit scale/min pair `j` out of a Q4_K superblock's
/// 12-byte `scales` array. Re-derived from `get_scale_min_k4` in llama.cpp
/// `ggml-quants.c`
/// <https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-quants.c>
/// (the same helper `dequantize_row_q4_K` and candle's `BlockQ4K::to_float`
/// use), duplicated here so the test decodes the file without depending on
/// the encoder internals it verifies.
fn q4_k_scale_min(j: usize, scales: &[u8; 12]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0f) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

/// Decodes one 144-byte Q4_K superblock (fp16 `d`/`dmin`, 12 packed scale
/// bytes, 128 quant bytes) into its 256 f32 values. Re-derived from
/// `dequantize_row_q4_K` in llama.cpp
/// <https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-quants.c>
/// and kept operation-for-operation identical to candle's
/// `BlockQ4K::to_float`, so the candle cross-check test compares two
/// independent implementations of the reference dequantizer bitwise.
fn decode_q4_k(block: &[u8]) -> [f32; 256] {
    assert_eq!(block.len(), 144);
    let d = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = half::f16::from_le_bytes([block[2], block[3]]).to_f32();
    let mut scales = [0u8; 12];
    scales.copy_from_slice(&block[4..16]);
    let qs = &block[16..144];
    let mut out = [0.0f32; 256];
    let mut is = 0;
    for step in (0..256).step_by(64) {
        let q = &qs[step / 2..step / 2 + 32];
        let (sc_lo, m_lo) = q4_k_scale_min(is, &scales);
        let (sc_hi, m_hi) = q4_k_scale_min(is + 1, &scales);
        let scale_lo = d * f32::from(sc_lo);
        let min_lo = dmin * f32::from(m_lo);
        let scale_hi = d * f32::from(sc_hi);
        let min_hi = dmin * f32::from(m_hi);
        for (i, &packed) in q.iter().enumerate() {
            out[step + i] = scale_lo * f32::from(packed & 0x0f) - min_lo;
        }
        for (i, &packed) in q.iter().enumerate() {
            out[step + 32 + i] = scale_hi * f32::from(packed >> 4) - min_hi;
        }
        is += 2;
    }
    out
}

#[test]
fn test_gguf_q4_0_roundtrip_metadata_block_bytes_and_error() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("q4_0.gguf");
    let exact: [f32; 32] = std::array::from_fn(|index| {
        if index < 16 {
            8.0 - index as f32
        } else {
            index as f32 - 23.0
        }
    });
    let scaled = exact.map(|value| value * 0.137);
    let fractional: [f32; 32] = std::array::from_fn(|index| index as f32 * 16.0 / 31.0 - 8.0);
    let weights: Vec<f32> = exact
        .into_iter()
        .chain(scaled)
        .chain([0.0; 32])
        .chain(fractional)
        .collect();
    let mut layer = Linear::<s![32, 4], CpuBackendImpl>::build(()).unwrap();
    set_export_weights(&mut layer, &weights);
    GgufExporter::<CpuBackendImpl, _>::from_module(&layer)
        .with_quantization(QuantScheme::W4A16_Q4_0)
        .save(&path)
        .unwrap();

    let info = inspect_file(&path).unwrap();
    assert_eq!(info.format, "GGUF v3");
    assert_eq!(info.tensor_count, 2);
    let weight = info
        .tensors
        .iter()
        .find(|tensor| tensor.name == "weight")
        .unwrap();
    assert_eq!(weight.dtype, "Q4_0");
    assert_eq!(weight.shape, vec![4, 32]);
    let bytes = std::fs::read(&path).unwrap();
    let (metadata, headers, data_start) = read_gguf(&bytes);
    assert_eq!(metadata["general.file_type"], 2);
    assert_eq!(metadata["general.alignment"], 32);
    assert_eq!(metadata["general.architecture"], "custom");
    let header = headers
        .iter()
        .find(|header| header.name == "weight")
        .unwrap();
    assert_eq!(header.dtype, 2);
    assert_eq!(header.dimensions, vec![32, 4]);
    let payload_start = data_start + header.offset;
    let payload = &bytes[payload_start..payload_start + 4 * 18];
    assert_eq!(bytes.len(), payload_start + 96);
    assert!(
        bytes[payload_start + 4 * 18..]
            .iter()
            .all(|&byte| byte == 0)
    );
    assert_eq!(
        &payload[..18],
        &[
            0x00, 0xbc, 0xf0, 0xe1, 0xd2, 0xc3, 0xb4, 0xa5, 0x96, 0x87, 0x78, 0x69, 0x5a, 0x4b,
            0x3c, 0x2d, 0x1e, 0x0f,
        ]
    );
    assert_eq!(decode_q4_0(&payload[..18]), exact);
    assert_eq!(&payload[20..36], &payload[2..18]);
    for (actual, expected) in decode_q4_0(&payload[18..36]).iter().zip(scaled) {
        assert!((actual - expected).abs() <= 8.0 * 0.137 / 64.0);
    }
    assert_eq!(&payload[36..38], &[0x00, 0x80]);
    assert_eq!(&payload[38..54], &[0x88; 16]);
    for (block, original) in payload.chunks_exact(18).zip(weights.chunks_exact(32)) {
        let signed_max = original.iter().copied().fold(0.0f32, |max, value| {
            if value.abs() > max.abs() { value } else { max }
        });
        let expected_scale = signed_max / -8.0;
        let scale = half::f16::from_le_bytes([block[0], block[1]]).to_f32();
        let scale_error = (scale - expected_scale).abs();
        assert!(scale_error <= expected_scale.abs() / 64.0);
        for (actual, expected) in decode_q4_0(block).iter().zip(original) {
            assert!((actual - expected).abs() <= expected_scale.abs() + 8.0 * scale_error);
        }
    }
}

/// Four Q4_K superblocks (one per weight row): a constant -12 (stage-1
/// hits max == min on every sub-block), an all-zero superblock, a
/// -32..31.75 ramp, and a constant 0.5. Used by the golden-byte,
/// round-trip and candle cross-check tests so they all encode the same
/// input.
fn q4_k_design_weights() -> [f32; 1024] {
    std::array::from_fn(|index| match index / 256 {
        0 => -12.0,
        1 => 0.0,
        2 => index as f32 % 256.0 * 0.25 - 32.0,
        _ => 0.5,
    })
}

/// Q4_K export end-to-end: file metadata (`general.file_type` 15 =
/// `LLAMA_FTYPE_MOSTLY_Q4_K_M`), tensor table (dtype 12, reversed dims
/// `[256, 4]`, 32-byte alignment), hand-derived golden superblock bytes
/// for the three constant/degenerate cases, and reference-dequantizer
/// round-trip bounds for all four. The golden bytes follow
/// `quantize_row_q4_K_ref` in llama.cpp
/// <https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-quants.c>.
#[test]
fn test_gguf_q4_k_roundtrip_metadata_golden_bytes_and_error() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("q4_k.gguf");
    let weights = q4_k_design_weights();
    let mut layer = Linear::<s![256, 4], CpuBackendImpl>::build(()).unwrap();
    set_export_weights(&mut layer, &weights);
    GgufExporter::<CpuBackendImpl, _>::from_module(&layer)
        .with_quantization(QuantScheme::W4A16_Q4_K_M)
        .save(&path)
        .unwrap();

    let info = inspect_file(&path).unwrap();
    assert_eq!(info.format, "GGUF v3");
    assert_eq!(info.tensor_count, 2);
    let weight = info
        .tensors
        .iter()
        .find(|tensor| tensor.name == "weight")
        .unwrap();
    assert_eq!(weight.dtype, "Q4_K");
    assert_eq!(weight.shape, vec![4, 256]);
    let bias = info
        .tensors
        .iter()
        .find(|tensor| tensor.name == "bias")
        .unwrap();
    assert_eq!(bias.dtype, "F32");
    assert_eq!(bias.shape, vec![4]);

    let bytes = std::fs::read(&path).unwrap();
    let (metadata, headers, data_start) = read_gguf(&bytes);
    assert_eq!(metadata["general.file_type"], 15);
    assert_eq!(metadata["general.alignment"], 32);
    assert_eq!(metadata["general.architecture"], "custom");
    let weight_header = headers
        .iter()
        .find(|header| header.name == "weight")
        .unwrap();
    assert_eq!(weight_header.dtype, 12);
    assert_eq!(weight_header.dimensions, vec![256, 4]);
    let bias_header = headers.iter().find(|header| header.name == "bias").unwrap();
    assert_eq!(bias_header.dtype, 0);
    assert_eq!(bias_header.offset, 0);
    // bias (4 f32 = 16 bytes) is padded to the 32-byte alignment, so the
    // weight payload starts at offset 32.
    assert_eq!(weight_header.offset, 32);
    assert!(
        bytes[data_start + 16..data_start + 32]
            .iter()
            .all(|&byte| byte == 0)
    );

    let payload_start = data_start + weight_header.offset;
    assert_eq!(bytes.len(), payload_start + 4 * 144);
    let payload = &bytes[payload_start..payload_start + 4 * 144];

    // Superblock 0 (constant -12.0): every sub-block degenerates to
    // (scale, min) = (0, 12), so d packs to 0, dmin to fp16(12/63), the
    // scale bytes to the max-min grid (ls = 0, lm = 63 in each 6-bit
    // slot) and all quants to 0.
    let mut expected_sb0 = Vec::new();
    expected_sb0.extend_from_slice(&0x0000u16.to_le_bytes());
    expected_sb0.extend_from_slice(&half::f16::from_f32(12.0 / 63.0).to_le_bytes());
    expected_sb0.extend_from_slice(&[
        0x00, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF, 0xF0, 0xF0, 0xF0, 0xF0,
    ]);
    expected_sb0.extend_from_slice(&[0u8; 128]);
    assert_eq!(&payload[0..144], expected_sb0.as_slice());

    // Superblock 1 (all zeros) is an all-zero block.
    assert_eq!(&payload[144..288], &[0u8; 144][..]);

    // Superblock 3 (constant 0.5): stage-1 (scale, min) = (1/30, 0),
    // ls = 63, lm = 0, quants all 15; d packs to fp16((1/30)/63).
    let mut expected_sb3 = Vec::new();
    expected_sb3.extend_from_slice(&half::f16::from_f32((1.0f32 / 30.0) / 63.0).to_le_bytes());
    expected_sb3.extend_from_slice(&0x0000u16.to_le_bytes());
    expected_sb3.extend_from_slice(&[
        0xFF, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00, 0x00, 0x0F, 0x0F, 0x0F, 0x0F,
    ]);
    expected_sb3.extend_from_slice(&[0xFFu8; 128]);
    assert_eq!(&payload[432..576], expected_sb3.as_slice());

    let dec0 = decode_q4_k(&payload[0..144]);
    assert!(dec0.iter().all(|value| (value + 12.0).abs() <= 0.01));
    let dec1 = decode_q4_k(&payload[144..288]);
    assert!(dec1.iter().all(|value| *value == 0.0));
    let dec3 = decode_q4_k(&payload[432..576]);
    assert!(dec3.iter().all(|value| (value - 0.5).abs() <= 0.001));

    // Superblock 2 (the ramp): positive-only sub-blocks clamp stage-1's
    // min to zero, so their tops quantize worst; the largest error is at
    // the ramp top (index 255: 1.9015), bound 2.0.
    let dec2 = decode_q4_k(&payload[288..432]);
    for (actual, expected) in dec2.iter().zip(&weights[512..766]) {
        assert!((actual - expected).abs() <= 2.0, "{actual} vs {expected}");
    }
}

#[test]
fn test_gguf_q4_0_mixed_f32_offsets_and_alignment() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("mixed.gguf");
    let mut model = Sequential(
        Linear::<s![32, 3], CpuBackendImpl>::build(()).unwrap(),
        Linear::<s![32, 3], CpuBackendImpl>::build(()).unwrap(),
    );
    set_export_weights(&mut model, &[1.0; 96]);
    let snapshot = collect_state::<CpuBackendImpl, _>(&model).unwrap();
    GgufExporter::<CpuBackendImpl, _>::from_module(&model)
        .with_quantization(QuantScheme::W4A16_Q4_0)
        .save(&path)
        .unwrap();
    let info = inspect_file(&path).unwrap();
    assert_eq!(info.tensor_count, 4);
    let bytes = std::fs::read(&path).unwrap();
    let (_, headers, data_start) = read_gguf(&bytes);
    assert_eq!(
        headers
            .iter()
            .map(|header| header.offset)
            .collect::<Vec<_>>(),
        vec![0, 32, 96, 128]
    );
    for (index, (header, (name, value))) in headers.iter().zip(snapshot.iter()).enumerate() {
        assert_eq!(header.name, name.as_str());
        let start = data_start + header.offset;
        let len = if header.name.ends_with("weight") {
            assert_eq!(header.dtype, 2);
            assert_eq!(info.tensors[index].dtype, "Q4_0");
            for block in bytes[start..start + 54].chunks_exact(18) {
                assert_eq!(decode_q4_0(block), [1.0; 32]);
            }
            54
        } else {
            assert_eq!(header.dtype, 0);
            assert_eq!(info.tensors[index].dtype, "F32");
            assert_eq!(&bytes[start..start + 12], value.bytes());
            12
        };
        let end = headers
            .get(index + 1)
            .map_or(bytes.len(), |next| data_start + next.offset);
        assert!(bytes[start + len..end].iter().all(|&byte| byte == 0));
    }
    assert_eq!(bytes.len(), data_start + 128 + 64);

    use candle_core::Device;
    use candle_core::quantized::{GgmlDType, gguf_file::Content};

    let mut reader = Cursor::new(&bytes);
    let content = Content::read(&mut reader).unwrap();
    assert_eq!(content.tensor_infos.len(), 4);
    assert_eq!(snapshot.len(), 4);
    for (name, value) in snapshot.iter() {
        let name = name.as_str();
        let (expected_shape, expected_dtype) = if name.ends_with("weight") {
            (&[3, 32][..], GgmlDType::Q4_0)
        } else {
            (&[3][..], GgmlDType::F32)
        };
        let tensor_info = content.tensor_infos.get(name).unwrap();
        assert_eq!(tensor_info.shape.dims(), expected_shape, "{name}");
        assert_eq!(tensor_info.ggml_dtype, expected_dtype, "{name}");
        let tensor = content.tensor(&mut reader, name, &Device::Cpu).unwrap();
        assert_eq!(tensor.shape().dims(), expected_shape, "{name}");
        assert_eq!(tensor.dtype(), expected_dtype, "{name}");
        let dequantized = tensor.dequantize(&Device::Cpu).unwrap();
        assert_eq!(dequantized.dims(), value.shape().dims(), "{name}");
        assert_eq!(dequantized.dtype(), candle_core::DType::F32, "{name}");
        let actual = dequantized.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let expected: Vec<f32> = value
            .bytes()
            .chunks_exact(4)
            .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
            .collect();
        assert_eq!(actual, expected, "{name}");
    }
}

#[test]
fn test_gguf_q4_0_payload_matches_independent_candle_writer() {
    use candle_core::quantized::{GgmlDType, QTensor, gguf_file};
    use candle_core::{Device, Tensor};

    let dir = tempdir().unwrap();
    let candle_path = dir.path().join("candle.gguf");
    let incin_path = dir.path().join("incin.gguf");
    let mut weights = Vec::with_capacity(8 * 32);
    for first in [-8.0, 8.0] {
        let mut block = [0.25; 32];
        block[0] = first;
        block[1] = -first;
        weights.extend(block);
    }
    for sign in [-1.0, 1.0] {
        for perturbation in [-0.0001, 0.0, 0.0001] {
            let mut block: [f32; 32] =
                std::array::from_fn(|index| sign * ((index % 16) as f32 - 7.5 + perturbation));
            block[0] = sign * -8.0;
            weights.extend(block);
        }
    }
    assert_eq!(weights.len(), 8 * 32);
    assert!(
        weights
            .iter()
            .all(|value| value.is_finite() && *value != 0.0)
    );
    let source = Tensor::from_slice(&weights, (8, 32), &Device::Cpu).unwrap();
    let quantized = QTensor::quantize(&source, GgmlDType::Q4_0).unwrap();
    let bias_source = Tensor::from_slice(&[0.0f32; 8], 8, &Device::Cpu).unwrap();
    let bias = QTensor::quantize(&bias_source, GgmlDType::F32).unwrap();
    gguf_file::write(
        &mut std::fs::File::create(&candle_path).unwrap(),
        &[],
        &[("bias", &bias), ("weight", &quantized)],
    )
    .unwrap();

    let mut layer = Linear::<s![32, 8], CpuBackendImpl>::build(()).unwrap();
    set_export_weights(&mut layer, &weights);
    GgufExporter::<CpuBackendImpl, _>::from_module(&layer)
        .with_quantization(QuantScheme::W4A16_Q4_0)
        .save(&incin_path)
        .unwrap();

    let mut payloads = Vec::new();
    for path in [&candle_path, &incin_path] {
        let bytes = std::fs::read(path).unwrap();
        let mut reader = Cursor::new(&bytes);
        let content = gguf_file::Content::read(&mut reader).unwrap();
        assert_eq!(content.tensor_infos.len(), 2);
        let info = content.tensor_infos.get("weight").unwrap();
        assert_eq!(info.ggml_dtype, GgmlDType::Q4_0);
        assert_eq!(info.shape.dims(), &[8, 32]);
        assert!(info.offset > 0);
        let start = usize::try_from(content.tensor_data_offset + info.offset).unwrap();
        let len =
            info.shape.elem_count() / info.ggml_dtype.block_size() * info.ggml_dtype.type_size();
        assert_eq!(len, 8 * 18);
        payloads.push(bytes[start..start + len].to_vec());
        let tensor = content.tensor(&mut reader, "weight", &Device::Cpu).unwrap();
        assert_eq!(tensor.dtype(), GgmlDType::Q4_0);
        assert_eq!(tensor.shape().dims(), &[8, 32]);
    }
    assert_eq!(payloads[0], payloads[1]);
}

/// Honest byte-compare verdict against candle's independent Q4_K writer on
/// ramp data: the payloads do NOT match. candle's `BlockQ4K::from_float`
/// runs `make_qkx1_quants` (unweighted, 5 refinement tries) as its first
/// stage instead of llama.cpp's `quantize_row_q4_K_ref` ->
/// `make_qkx2_quants` (weights `av_x + |x|`, `rmin = -1`, `rdelta = 0.1`,
/// `nstep = 20`) and rounds ties away from zero (`f32::round`) instead of
/// ggml's `nearest_int`, so equal encodings are not achievable. Both
/// payloads remain valid Q4_K blocks of the same ramp (bounded by the
/// reference dequantizer below), and `ggml-quants` 0.1.0 cannot substitute
/// as a byte oracle because its `Q4K::quantize` is `todo!()` (pinned by
/// `test_ggml_quants_q4k_quantize_is_unimplemented`).
#[test]
fn test_gguf_q4_k_payload_differs_from_candle_writer() {
    use candle_core::quantized::{GgmlDType, QTensor, gguf_file};
    use candle_core::{Device, Tensor};

    let ramp: [f32; 256] = std::array::from_fn(|index| index as f32 * 0.25 - 32.0);
    let dir = tempdir().unwrap();
    let candle_path = dir.path().join("candle.gguf");
    let source = Tensor::from_slice(&ramp, 256, &Device::Cpu).unwrap();
    let quantized = QTensor::quantize(&source, GgmlDType::Q4K).unwrap();
    gguf_file::write(
        &mut std::fs::File::create(&candle_path).unwrap(),
        &[],
        &[("weight", &quantized)],
    )
    .unwrap();
    let candle_bytes = std::fs::read(&candle_path).unwrap();
    let mut reader = Cursor::new(&candle_bytes);
    let content = gguf_file::Content::read(&mut reader).unwrap();
    let info = content.tensor_infos.get("weight").unwrap();
    assert_eq!(info.ggml_dtype, GgmlDType::Q4K);
    assert_eq!(info.shape.dims(), &[256]);
    let start = usize::try_from(content.tensor_data_offset + info.offset).unwrap();
    let len = info.shape.elem_count() / info.ggml_dtype.block_size() * info.ggml_dtype.type_size();
    assert_eq!(len, 144);
    let candle_payload = &candle_bytes[start..start + len];

    let (dtype, incin_payload) = export_values(&[256], &ramp, QuantScheme::W4A16_Q4_K_M);
    assert_eq!(dtype, 12);
    assert_ne!(candle_payload, incin_payload.as_slice());

    for payload in [candle_payload, &incin_payload[..]] {
        for (actual, expected) in decode_q4_k(payload).iter().zip(&ramp) {
            assert!((actual - expected).abs() <= 2.0, "{actual} vs {expected}");
        }
    }
}

/// Independent-decoder cross-check: candle's `BlockQ4K::to_float` is a
/// faithful port of llama.cpp's `dequantize_row_q4_K` (k_quants.c L735,
/// <https://github.com/ggerganov/llama.cpp/blob/master/ggml/src/ggml-quants.c>),
/// so its dequantized tensor must match this file's re-derived
/// `decode_q4_k` bitwise on every payload element.
#[test]
fn test_gguf_q4_k_dequantization_matches_candle_reference() {
    use candle_core::Device;
    use candle_core::quantized::{GgmlDType, gguf_file::Content};

    let dir = tempdir().unwrap();
    let path = dir.path().join("q4_k.gguf");
    let weights = q4_k_design_weights();
    let mut layer = Linear::<s![256, 4], CpuBackendImpl>::build(()).unwrap();
    set_export_weights(&mut layer, &weights);
    GgufExporter::<CpuBackendImpl, _>::from_module(&layer)
        .with_quantization(QuantScheme::W4A16_Q4_K_M)
        .save(&path)
        .unwrap();

    let bytes = std::fs::read(&path).unwrap();
    let (_, headers, data_start) = read_gguf(&bytes);
    let weight_header = headers
        .iter()
        .find(|header| header.name == "weight")
        .unwrap();
    let payload = &bytes[data_start + weight_header.offset..][..4 * 144];

    let mut reader = Cursor::new(&bytes);
    let content = Content::read(&mut reader).unwrap();
    assert_eq!(content.tensor_infos.len(), 2);
    let info = content.tensor_infos.get("weight").unwrap();
    assert_eq!(info.ggml_dtype, GgmlDType::Q4K);
    assert_eq!(info.shape.dims(), &[4, 256]);
    let tensor = content.tensor(&mut reader, "weight", &Device::Cpu).unwrap();
    let dequantized = tensor
        .dequantize(&Device::Cpu)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap();
    assert_eq!(dequantized.len(), 4 * 256);
    for (superblock, block) in payload.chunks_exact(144).enumerate() {
        for (index, expected) in decode_q4_k(block).iter().enumerate() {
            let actual = dequantized[superblock * 256 + index];
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "superblock {superblock} element {index}"
            );
        }
    }
}

#[test]
fn test_gguf_q4_0_empty_tensors_stay_f32() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("empty.gguf");
    let layer = Linear::<s![0, 3], CpuBackendImpl>::build(()).unwrap();
    GgufExporter::<CpuBackendImpl, _>::from_module(&layer)
        .with_quantization(QuantScheme::W4A16_Q4_0)
        .save(&path)
        .unwrap();
    let info = inspect_file(&path).unwrap();
    assert_eq!(info.tensor_count, 2);
    assert!(info.tensors.iter().all(|tensor| tensor.dtype == "F32"));
    assert!(info.tensors.iter().any(|tensor| tensor.shape == [3, 0]));
}

fn export_values(shape: &[usize], values: &[f32], scheme: QuantScheme) -> (u32, Vec<u8>) {
    let dir = tempdir().unwrap();
    let path = dir.path().join("values.gguf");
    let mut param = Param::<Dyn, CpuBackendImpl>::ones(shape.to_vec()).unwrap();
    set_export_weights(&mut param, values);
    GgufExporter::<CpuBackendImpl, _>::from_module(&param)
        .with_quantization(scheme)
        .save(&path)
        .unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let (_, headers, data_start) = read_gguf(&bytes);
    assert_eq!(headers.len(), 1);
    let header = &headers[0];
    assert_eq!(header.offset, 0);
    let expected_dims: Vec<u64> = if shape.is_empty() {
        vec![1]
    } else {
        shape.iter().rev().map(|&dim| dim as u64).collect()
    };
    assert_eq!(header.dimensions, expected_dims);
    let len = match header.dtype {
        0 => values.len() * 4,
        2 => values.len() / 32 * 18,
        8 => values.len() / 32 * 34,
        12 => values.len() / 256 * 144,
        dtype => panic!("unexpected tensor type {dtype}"),
    };
    assert_eq!(bytes.len(), data_start + len.next_multiple_of(32));
    assert!(bytes[data_start + len..].iter().all(|&byte| byte == 0));
    let info = inspect_file(&path).unwrap();
    assert_eq!(info.tensor_count, 1);
    assert_eq!(
        info.tensors[0].dtype,
        match header.dtype {
            0 => "F32",
            2 => "Q4_0",
            8 => "Q8_0",
            12 => "Q4_K",
            _ => unreachable!(),
        }
    );
    (header.dtype, bytes[data_start..data_start + len].to_vec())
}

fn reference_q4_0(values: &[f32; 32]) -> ([u8; 2], [u8; 16]) {
    use ggml_quants::{Q4_0, Quantize};

    let reference = Q4_0::quantize(values);
    (reference.delta.to_le_bytes(), reference.quants)
}

#[test]
fn test_gguf_q4_0_signed_max_ties_zero_and_reference_rounding() {
    let (dtype, bytes) = export_values(&[32], &[-8.0; 32], QuantScheme::W4A16_Q4_0);
    assert_eq!(dtype, 2);
    assert_eq!(&bytes[..2], &[0x00, 0x3c]);
    assert_eq!(&bytes[2..], &[0x00; 16]);
    assert_eq!(decode_q4_0(&bytes), [-8.0; 32]);

    for first in [-8.0, 8.0] {
        let mut values = [0.0; 32];
        values[0] = first;
        values[1] = -first;
        let (_, bytes) = export_values(&[32], &values, QuantScheme::W4A16_Q4_0);
        assert_eq!(
            &bytes[..2],
            if first < 0.0 {
                &[0x00, 0x3c]
            } else {
                &[0x00, 0xbc]
            }
        );
        assert_eq!(&bytes[2..4], &[0x80, 0x8f]);
        assert_eq!(&bytes[4..], &[0x88; 14]);
        let (scale_bytes, packed) = reference_q4_0(&values);
        assert_eq!(&bytes[..2], &scale_bytes);
        assert_eq!(&bytes[2..], &packed);
    }

    for zero in [0.0, -0.0] {
        let (_, bytes) = export_values(&[32], &[zero; 32], QuantScheme::W4A16_Q4_0);
        assert_eq!(&bytes[..2], &[0x00, 0x80]);
        assert_eq!(&bytes[2..], &[0x88; 16]);
    }

    for sign in [-1.0, 1.0] {
        for perturbation in [-0.0001, 0.0, 0.0001] {
            let mut values: [f32; 32] =
                std::array::from_fn(|index| sign * ((index % 16) as f32 - 7.5 + perturbation));
            values[0] = sign * -8.0;
            let (_, bytes) = export_values(&[32], &values, QuantScheme::W4A16_Q4_0);
            let (scale_bytes, packed) = reference_q4_0(&values);
            assert_eq!(&bytes[..2], &scale_bytes);
            assert_eq!(&bytes[2..], &packed);
            if perturbation == 0.0 {
                assert_eq!(
                    &bytes[2..],
                    &[
                        0x10, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb, 0xcc,
                        0xdd, 0xee, 0xff, 0xff,
                    ]
                );
            }
        }
    }
}

/// `ggml-quants` 0.1.0 is the only other Rust crate exposing the GGML
/// `Q4K` block type, but its `quantize`/`dequantize` are `todo!()`, so it
/// cannot serve as a byte-compare oracle for our encoder. This pins that
/// fact: if a future release implements the reference quantizer the test
/// stops panicking and should be replaced by a real byte-compare against
/// `Q4K::quantize`.
#[test]
#[should_panic(expected = "not yet implemented")]
fn test_ggml_quants_q4k_quantize_is_unimplemented() {
    use ggml_quants::{Q4K, Quantize};
    let _ = Q4K::quantize(&[0.25f32; 256]);
}

#[test]
fn test_gguf_block_quantization_requires_complete_rows() {
    for scheme in [QuantScheme::Q8_0, QuantScheme::W4A16_Q4_0] {
        for shape in [&[8, 4][..], &[2, 16], &[0, 32], &[32, 0]] {
            let numel = shape.iter().product();
            let values: Vec<f32> = (0..numel).map(|index| index as f32 * 0.25 - 3.0).collect();
            let (dtype, bytes) = export_values(shape, &values, scheme);
            assert_eq!(dtype, 0);
            assert_eq!(
                bytes,
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>()
            );
        }
        for shape in [&[4, 32][..], &[32]] {
            let values = vec![1.0; shape.iter().product()];
            let (dtype, _) = export_values(shape, &values, scheme);
            assert_eq!(dtype, scheme.ggml_type_id());
        }
    }
}

/// Q4_K packs 256-element superblocks along the last dimension, so shapes
/// whose last dim is not a multiple of 256 stay F32 — including shapes
/// that Q4_0 would accept (e.g. `[4, 32]`).
#[test]
fn test_gguf_q4_k_requires_superblock_sized_rows() {
    for shape in [
        &[8, 128][..],
        &[16, 16],
        &[4, 32],
        &[2, 128],
        &[0, 256],
        &[256, 0],
    ] {
        let numel = shape.iter().product();
        let values: Vec<f32> = (0..numel).map(|index| index as f32 * 0.25 - 3.0).collect();
        let (dtype, bytes) = export_values(shape, &values, QuantScheme::W4A16_Q4_K_M);
        assert_eq!(dtype, 0, "shape {shape:?} must stay F32");
        assert_eq!(
            bytes,
            values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect::<Vec<_>>()
        );
    }
    for shape in [&[256][..], &[4, 256], &[1, 1, 1, 256], &[3, 256]] {
        let values = vec![1.0f32; shape.iter().product()];
        let (dtype, _) = export_values(shape, &values, QuantScheme::W4A16_Q4_K_M);
        assert_eq!(
            dtype,
            QuantScheme::W4A16_Q4_K_M.ggml_type_id(),
            "shape {shape:?}"
        );
    }
}

#[test]
fn test_gguf_q4_0_numeric_domain_falls_back_whole_tensor() {
    let half_min = half::f16::from_bits(1).to_f32();
    for invalid in [
        f32::from_bits(0x7fc01234),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        -f32::MAX,
        65520.0 * 8.0,
        -65520.0 * 8.0,
        half_min * 4.0,
        -half_min * 4.0,
        f32::MIN_POSITIVE,
        f32::from_bits(8),
        f32::from_bits(1),
    ] {
        for bad_block in [0, 1] {
            let mut values = [1.0; 64];
            values[bad_block * 32..bad_block * 32 + 32].fill(invalid);
            let (dtype, bytes) = export_values(&[2, 32], &values, QuantScheme::W4A16_Q4_0);
            assert_eq!(dtype, 0, "invalid value {invalid:?}");
            assert_eq!(
                bytes,
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>()
            );
        }
    }
    for value in [
        half_min * 8.0,
        -half_min * 8.0,
        65504.0 * 8.0,
        -65504.0 * 8.0,
    ] {
        let (dtype, bytes) = export_values(&[32], &[value; 32], QuantScheme::W4A16_Q4_0);
        assert_eq!(dtype, 2);
        assert_eq!(decode_q4_0(&bytes), [value; 32]);
    }
}

/// Whole-tensor F32 fallback for values outside the representable Q4_K
/// grid. The domain differs from Q4_0: `65520 * 8` fits (its `d` packs to
/// `max/63`) while `half_min * 8` falls back (`d` rounds to a zero/subnormal
/// fp16), and `1e8` is rejected where Q4_0's coarser grid accepts it —
/// any single bad element in either superblock must keep the entire
/// tensor at F32.
#[test]
fn test_gguf_q4_k_numeric_domain_falls_back_whole_tensor() {
    let half_min = half::f16::from_bits(1).to_f32();
    for invalid in [
        f32::from_bits(0x7fc01234),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        -f32::MAX,
        half_min * 4.0,
        -half_min * 4.0,
        half_min * 8.0,
        -half_min * 8.0,
        f32::MIN_POSITIVE,
        f32::from_bits(8),
        f32::from_bits(1),
        1e8,
        -1e8,
    ] {
        for bad_block in [0, 1] {
            let mut values = [1.0; 512];
            values[bad_block * 256..bad_block * 256 + 256].fill(invalid);
            let (dtype, bytes) = export_values(&[2, 256], &values, QuantScheme::W4A16_Q4_K_M);
            assert_eq!(dtype, 0, "invalid value {invalid:?}");
            assert_eq!(
                bytes,
                values
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect::<Vec<_>>()
            );
        }
    }
    for value in [
        65504.0 * 8.0,
        -65504.0 * 8.0,
        65520.0 * 8.0,
        -65520.0 * 8.0,
        0.001,
    ] {
        let (dtype, bytes) = export_values(&[256], &[value; 256], QuantScheme::W4A16_Q4_K_M);
        assert_eq!(dtype, 12);
        for actual in decode_q4_k(&bytes) {
            assert!(
                (actual - value).abs() <= value.abs() * 0.02,
                "{value}: {actual}"
            );
        }
    }
}

#[test]
fn test_gguf_scalar_normalization_and_rank_limit() {
    for scheme in [
        QuantScheme::F32,
        QuantScheme::Q8_0,
        QuantScheme::W4A16_Q4_0,
        QuantScheme::W4A16_Q4_K_M,
    ] {
        // Q4_K needs a 256-element superblock; the other schemes a
        // 32-element row.
        let width = if scheme == QuantScheme::W4A16_Q4_K_M {
            256
        } else {
            32
        };
        let (dtype, bytes) = export_values(&[], &[1.25], scheme);
        assert_eq!(dtype, 0);
        assert_eq!(bytes, 1.25f32.to_le_bytes());
        let values = vec![1.0f32; width];
        let (dtype, _) = export_values(&[1, 1, 1, width], &values, scheme);
        assert_eq!(dtype, scheme.ggml_type_id());

        let dir = tempdir().unwrap();
        let path = dir.path().join("rank5.gguf");
        let param = Param::<Dyn, CpuBackendImpl>::ones(vec![1, 1, 1, 1, width]).unwrap();
        let exporter =
            GgufExporter::<CpuBackendImpl, _>::from_module(&param).with_quantization(scheme);
        let error = exporter.save(&path).unwrap_err();
        assert!(matches!(
            error,
            incin_core::error::Error::InvalidModuleState {
                operation: "GGUF export",
                ..
            }
        ));
        assert!(!path.exists());
        std::fs::write(&path, b"preserve existing file").unwrap();
        assert!(exporter.save(&path).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), b"preserve existing file");
    }
}

#[test]
fn test_gguf_export_and_inspect_roundtrip() {
    let dir = tempdir().unwrap();
    let gguf_path = dir.path().join("model.gguf");

    let layer = Linear::<s![32, 8], CpuBackendImpl>::build(()).unwrap();

    GgufExporter::<CpuBackendImpl, _>::from_module(&layer)
        .with_quantization(QuantScheme::Q8_0)
        .save(&gguf_path)
        .unwrap();

    assert!(gguf_path.exists());

    let info = inspect_file(&gguf_path).unwrap();
    assert!(info.format.contains("GGUF"));
    assert_eq!(info.tensor_count, 2); // weight and bias
}

/// Regression test: the exporter used to label every tensor's `ggml_type`
/// with the requested `QuantScheme` while always writing raw float bytes,
/// so a "Q8_0" file was actually full-precision data mislabeled as
/// quantized. Weight is [8, 32] = 256 elements (eight Q8_0 blocks: 272 bytes),
/// bias is [8] elements (not a multiple of 32, so it must stay F32: 32
/// bytes) - this pins both the per-tensor quantization eligibility rule
/// and that the declared dtype in the tensor table matches the bytes
/// actually written.
#[test]
fn test_gguf_export_actually_quantizes_eligible_tensors_and_leaves_others_at_f32() {
    let dir = tempdir().unwrap();
    let gguf_path = dir.path().join("model.gguf");

    let layer = Linear::<s![32, 8], CpuBackendImpl>::build(()).unwrap();

    GgufExporter::<CpuBackendImpl, _>::from_module(&layer)
        .with_quantization(QuantScheme::Q8_0)
        .save(&gguf_path)
        .unwrap();

    let info = inspect_file(&gguf_path).unwrap();
    assert_eq!(info.tensors.len(), 2);

    let weight = info
        .tensors
        .iter()
        .find(|t| t.name.contains("weight"))
        .unwrap();
    assert_eq!(weight.dtype, "Q8_0");
    assert_eq!(weight.shape, vec![8, 32]);

    let bias = info
        .tensors
        .iter()
        .find(|t| t.name.contains("bias"))
        .unwrap();
    assert_eq!(bias.dtype, "F32");
    assert_eq!(bias.shape, vec![8]);
}

/// `QuantScheme`s the backend can't actually convert to (yet) must fail
/// loudly at export time rather than silently writing float bytes under a
/// mismatching `ggml_type` header.
#[test]
fn test_gguf_export_rejects_unimplemented_quant_schemes() {
    let dir = tempdir().unwrap();
    let gguf_path = dir.path().join("model.gguf");

    let layer = Linear::<s![32, 8], CpuBackendImpl>::build(()).unwrap();

    let result = GgufExporter::<CpuBackendImpl, _>::from_module(&layer)
        .with_quantization(QuantScheme::F16)
        .save(&gguf_path);

    assert!(result.is_err());
}

#[test]
fn test_mlx_export_roundtrip() {
    let dir = tempdir().unwrap();
    let mlx_dir = dir.path().join("mlx_model");

    let layer = Linear::<s![32, 8], CpuBackendImpl>::build(()).unwrap();
    let config = r#"{"model_type": "linear", "hidden_size": 8}"#;

    MlxExporter::export_dir::<CpuBackendImpl, _, _>(&layer, &mlx_dir, config).unwrap();

    assert!(mlx_dir.join("weights.safetensors").exists());
    assert!(mlx_dir.join("config.json").exists());

    let info = inspect_file(mlx_dir.join("weights.safetensors")).unwrap();
    assert_eq!(info.format, "SafeTensors Checkpoint");
    assert_eq!(info.tensor_count, 2);
}
