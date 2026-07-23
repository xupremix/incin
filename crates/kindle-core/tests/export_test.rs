extern crate kindle_core as kindle;
use kindle_backends::cpu::CpuBackendImpl;
use kindle_core::io::{GgufExporter, MlxExporter, QuantScheme, inspect_file};
use kindle_core::prelude::*;
use tempfile::tempdir;

#[test]
fn test_gguf_export_and_inspect_roundtrip() {
    let dir = tempdir().unwrap();
    let gguf_path = dir.path().join("model.gguf");

    let layer = Linear::<s![4, 8], CpuBackendImpl>::build(()).unwrap();

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
/// quantized. Weight is [8, 4] = 32 elements (one Q8_0 block: 34 bytes),
/// bias is [8] elements (not a multiple of 32, so it must stay F32: 32
/// bytes) — this pins both the per-tensor quantization eligibility rule
/// and that the declared dtype in the tensor table matches the bytes
/// actually written.
#[test]
fn test_gguf_export_actually_quantizes_eligible_tensors_and_leaves_others_at_f32() {
    let dir = tempdir().unwrap();
    let gguf_path = dir.path().join("model.gguf");

    let layer = Linear::<s![4, 8], CpuBackendImpl>::build(()).unwrap();

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
    assert_eq!(weight.shape, vec![8, 4]);

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

    let layer = Linear::<s![4, 8], CpuBackendImpl>::build(()).unwrap();

    let result = GgufExporter::<CpuBackendImpl, _>::from_module(&layer)
        .with_quantization(QuantScheme::F16)
        .save(&gguf_path);

    assert!(result.is_err());
}

#[test]
fn test_mlx_export_roundtrip() {
    let dir = tempdir().unwrap();
    let mlx_dir = dir.path().join("mlx_model");

    let layer = Linear::<s![4, 8], CpuBackendImpl>::build(()).unwrap();
    let config = r#"{"model_type": "linear", "hidden_size": 8}"#;

    MlxExporter::export_dir::<CpuBackendImpl, _, _>(&layer, &mlx_dir, config).unwrap();

    assert!(mlx_dir.join("weights.safetensors").exists());
    assert!(mlx_dir.join("config.json").exists());

    let info = inspect_file(mlx_dir.join("weights.safetensors")).unwrap();
    assert_eq!(info.format, "SafeTensors Checkpoint");
    assert_eq!(info.tensor_count, 2);
}
