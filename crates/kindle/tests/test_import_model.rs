use kindle::prelude::*;
use kindle::candle::CandleBackend;

#[test]
fn test_onnx_parsing() {
    kindle_macros::import_model!("tests/basic_model.onnx", OnnxModel);
    
    // We generated the struct with PhantomData, so we can instantiate it
    let mut onnx = OnnxModel::<CandleBackend>::new();
    let _ = onnx.load_default_weights();
    
    // Test the generated forward function
    let dummy_input = kindle::prelude::Tensor::<kindle::prelude::Dyn, CandleBackend>::zeros(vec![1, 10]).unwrap();
    let out = onnx.forward(dummy_input).unwrap();
    assert_eq!(out.dims(), &[1, 5]);
}
