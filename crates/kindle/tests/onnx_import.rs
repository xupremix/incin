use kindle::prelude::*;

#[test]
/// Test onnx advanced ops.
fn test_onnx_advanced_ops() {
    import_model!("../../test_models/advanced.onnx", TestOps);

    /// Backend.
    type Backend = kindle_backends::cpu::CpuBackendImpl;

    let x = Tensor::<s![1, 3, 224, 224], Backend>::zeros(()).unwrap();
    let _shape = Tensor::<s![2], Backend>::zeros(()).unwrap();
    let model = TestOps {
        _shape: kindle::Param::zeros(()).unwrap(),
        _marker: std::marker::PhantomData,
    };

    let _out = model.forward(x).unwrap();
}

#[test]
/// Test onnx control flow if.
fn test_onnx_control_flow_if() {
    import_model!("../../test_models/if.onnx", TestIf);
    /// Backend.
    type Backend = kindle_backends::cpu::CpuBackendImpl;

    let cond = Tensor::<s![1], Backend>::zeros(()).unwrap();
    let x = Tensor::<s![1], Backend>::zeros(()).unwrap();
    let model = TestIf {
        _marker: std::marker::PhantomData,
    };

    // The if graph expects one output, which matches `forward` signature.
    let _out = model.forward(cond, x).unwrap();
}
