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

#[test]
fn test_gguf_scalar_normalization_and_rank_limit() {
    for scheme in [QuantScheme::F32, QuantScheme::Q8_0, QuantScheme::W4A16_Q4_0] {
        let (dtype, bytes) = export_values(&[], &[1.25], scheme);
        assert_eq!(dtype, 0);
        assert_eq!(bytes, 1.25f32.to_le_bytes());
        let (dtype, _) = export_values(&[1, 1, 1, 32], &[1.0; 32], scheme);
        assert_eq!(dtype, scheme.ggml_type_id());

        let dir = tempdir().unwrap();
        let path = dir.path().join("rank5.gguf");
        let param = Param::<Dyn, CpuBackendImpl>::ones(vec![1, 1, 1, 1, 32]).unwrap();
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
