use incin::prelude::*;

model!("resnet18.onnx", Resnet18);

fn main() {
    println!("ResNet18 successfully parsed into Rust AST!");

    // We can instantiate it
    let mut model = Resnet18::<DefaultBackend>::new();
    model.load_default_weights().unwrap();
}
