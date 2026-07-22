use kindle_backends::cpu::CpuBackendImpl;
use kindle_core::prelude::*;
use std::collections::BTreeMap;

struct TestModule<B: Backend> {
    weight: Tensor<Dyn, B>,
}

impl<B: Backend> StateDict<B> for TestModule<B> {
    fn load_state_dict(
        &mut self,
        _prefix: &str,
        tensors: &BTreeMap<String, Tensor<Dyn, B>>,
    ) -> Result<()> {
        if let Some(w) = tensors.get("weight") {
            self.weight = w.clone();
        }
        Ok(())
    }

    fn state_dict(&self, _prefix: &str, tensors: &mut BTreeMap<String, Tensor<Dyn, B>>) {
        tensors.insert("weight".to_string(), self.weight.clone());
    }
}

#[test]
fn test_safetensors_save_and_load_roundtrip_cpu() {
    type B = CpuBackendImpl;
    let initial_tensor: Tensor<Dyn, B> =
        Tensor::from_slice(&[1.5f32, 2.5, 3.5, 4.5], vec![2, 2]).unwrap();
    let module = TestModule {
        weight: initial_tensor,
    };

    let tmp_dir = std::env::temp_dir();
    let file_path = tmp_dir.join("kindle_test_cpu_weights.safetensors");

    save_safetensors(&module, &file_path).unwrap();
    assert!(file_path.exists());

    let fresh_tensor: Tensor<Dyn, B> = Tensor::zeros(vec![2, 2]).unwrap();
    let mut loaded_module = TestModule {
        weight: fresh_tensor,
    };

    load_safetensors(&mut loaded_module, &file_path).unwrap();

    let loaded_bytes = B::to_bytes::<f32>(loaded_module.weight.inner()).unwrap();
    let initial_bytes = B::to_bytes::<f32>(module.weight.inner()).unwrap();
    assert_eq!(loaded_bytes, initial_bytes);

    let _ = std::fs::remove_file(file_path);
}
