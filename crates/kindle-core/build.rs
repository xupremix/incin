use std::io::Result;

fn main() -> Result<()> {
    // Only rebuild if the .proto file changes
    println!("cargo:rerun-if-changed=proto/onnx.proto");

    let mut config = prost_build::Config::new();
    // Use BTreeMap instead of BTreeMap for deterministic builds and `no_std` compatibility
    config.btree_map(["."]);
    if let Err(e) = config.compile_protos(&["proto/onnx.proto"], &["proto/"]) {
        eprintln!(
            "\n================================================================================"
        );
        eprintln!(
            "Kindle Core Build Error: Failed to compile ONNX Protocol Buffers (proto/onnx.proto)."
        );
        eprintln!("Reason: {}", e);
        eprintln!(
            "================================================================================"
        );
        eprintln!("`protoc` (Protocol Buffers Compiler) is required to build kindle-core.");
        eprintln!("Please install `protoc` on your system:");
        eprintln!("  - Ubuntu / Debian: sudo apt-get install -y protobuf-compiler");
        eprintln!("  - macOS: brew install protobuf");
        eprintln!(
            "  - Windows / Manual: Set PROTOC environment variable pointing to protoc binary."
        );
        eprintln!(
            "================================================================================\n"
        );
        return Err(e);
    }
    Ok(())
}
