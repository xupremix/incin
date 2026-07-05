use kindle::prelude::*;

#[test]
fn test_onnx_advanced_ops() {
    import_model!("../../test_models/advanced.onnx", TestOps);

    type Backend = kindle_core::prelude::dummy::DummyBackend<f32, kindle_core::tensor::device::Cpu>;

    let x = Tensor::<(Const<1>, Const<3>, Const<224>, Const<224>), Backend>::zeros(()).unwrap();
    let _shape = Tensor::<(Const<2>,), Backend>::zeros(()).unwrap();
    let model = TestOps { 
        _shape: kindle::nn::Param::zeros(()).unwrap(),
        _marker: std::marker::PhantomData,
    };

    let _out = model.forward(x).unwrap();
}

#[test]
fn test_onnx_control_flow_if() {
    import_model!("../../test_models/if.onnx", TestIf);
    type Backend = NdarrayBackend<f32, kindle_core::tensor::device::Cpu>;

    let cond = Tensor::<(Const<1>,), Backend>::zeros(()).unwrap();
    let x = Tensor::<(Const<1>,), Backend>::zeros(()).unwrap();
    let model = TestIf {
        _marker: std::marker::PhantomData,
    };

    // The if graph expects one output, which matches `forward` signature.
    let _out = model.forward(cond, x).unwrap();
}
