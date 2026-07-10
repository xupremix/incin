use kindle::prelude::*;

import_model!("resnet18.onnx", Resnet18);

fn main() {
    println!("ResNet18 successfully parsed into Rust AST!");

    // We can instantiate it
    let mut model = Resnet18::<kindle_backends::dummy::DummyBackend<f32, Cpu>>::new();
    model.load_default_weights().unwrap();
}
