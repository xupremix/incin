use std::io::Result;

fn main() -> Result<()> {
    // Only rebuild if the .proto file changes
    println!("cargo:rerun-if-changed=proto/onnx.proto");
    
    let mut config = prost_build::Config::new();
    // Use BTreeMap instead of HashMap for deterministic builds and `no_std` compatibility
    config.btree_map(&["."]);
    // Configure prost to compile onnx.proto into the OUT_DIR
    config.compile_protos(&["proto/onnx.proto"], &["proto/"])?;
    Ok(())
}
