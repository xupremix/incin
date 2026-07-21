use kindle::prelude::*;

#[test]
/// Auto-generated documentation for test_onnx_advanced_ops.
fn test_onnx_advanced_ops() {
    import_model!("../../test_models/advanced.onnx", TestOps);

    /// Auto-generated documentation for Backend.
    type Backend = kindle_backends::cpu::CpuBackend<f32, kindle_core::prelude::Cpu>;

    let x = Tensor::<s![1, 3, 224, 224], Backend>::zeros(()).unwrap();
    let _shape = Tensor::<s![2], Backend>::zeros(()).unwrap();
    let model = TestOps {
        _shape: kindle::Param::zeros(()).unwrap(),
        _marker: std::marker::PhantomData,
    };

    let _out = model.forward(x).unwrap();
}

#[test]
/// Auto-generated documentation for test_onnx_control_flow_if.
fn test_onnx_control_flow_if() {
    import_model!("../../test_models/if.onnx", TestIf);
    /// Auto-generated documentation for Backend.
    type Backend = kindle_backends::cpu::CpuBackend<f32, kindle_core::prelude::Cpu>;

    let cond = Tensor::<s![1], Backend>::zeros(()).unwrap();
    let x = Tensor::<s![1], Backend>::zeros(()).unwrap();
    let model = TestIf {
        _marker: std::marker::PhantomData,
    };

    // The if graph expects one output, which matches `forward` signature.
    let _out = model.forward(cond, x).unwrap();
}
