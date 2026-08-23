//! Integration coverage for `test_safetensors_save_and_load_roundtrip_cpu` on the documented public surface.
#![cfg(feature = "cpu")]

extern crate incin_core as incin;

use incin_backends::cpu::CpuBackendImpl;
use incin_core::prelude::*;

#[test]
fn test_safetensors_save_and_load_roundtrip_cpu() {
    type B = CpuBackendImpl;
    let module = Linear::<s![2, 2], B>::build(()).unwrap();

    let file_path = std::env::temp_dir().join(format!(
        "incin_test_cpu_weights_{}.safetensors",
        std::process::id()
    ));

    module.save(Format::Safetensors, &file_path).unwrap();
    assert!(file_path.exists());

    let mut loaded_module = Linear::<s![2, 2], B>::build(()).unwrap();
    loaded_module.load(Format::Safetensors, &file_path).unwrap();
    assert_eq!(
        collect_state::<B, _>(&loaded_module).unwrap().len(),
        collect_state::<B, _>(&module).unwrap().len()
    );

    let _ = std::fs::remove_file(file_path);
}
