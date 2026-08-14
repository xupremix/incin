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

    save_safetensors(&module, &file_path).unwrap();
    assert!(file_path.exists());

    let mut loaded_module = Linear::<s![2, 2], B>::build(()).unwrap();
    load_safetensors(&mut loaded_module, &file_path).unwrap();
    assert_eq!(
        loaded_module.state_dict().unwrap().len(),
        module.state_dict().unwrap().len()
    );

    let _ = std::fs::remove_file(file_path);
}
