

// import_model! reads the safetensors file AT COMPILE TIME to build the nested module struct.
// For this example to compile, `resnet18.safetensors` must exist locally during compilation.
// In a real workflow, you would typically download it once in a build.rs script,
// or use `import_model!` on a generic ONNX/Safetensors file you have locally.
//
// Uncomment below when you have a valid Safetensors file:
//
// import_model!("resnet18.safetensors", ResNet18);
//
// fn main() -> Result<()> {
//     // Download weights from Hugging Face Hub (this will be fast if already cached!)
//     let path = kindle::hub::Api::new()?
//         .model("timm/resnet18.tv_in1k")
//         .get("model.safetensors")?;
//
//     println!("Weights available at: {:?}", path);
//
//     // Initialize our statically-generated struct
//     let mut model = ResNet18::<CandleBackend<f32, Cpu>>::new();
//
//     // Load the downloaded weights directly into the struct!
//     model.load_default_weights()?;
//
//     println!("Model successfully loaded from Hub!");
//     Ok(())
// }

fn main() {
    println!("See source code for Hub import example. You need a local .safetensors file for the macro to parse at compile time!");
}
