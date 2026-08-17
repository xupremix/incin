/// `onnx`.
///
/// The module body is `src/generated/onnx.rs`, a checked-in `prost-build`
/// output rather than a build-script product. Generating it at build time made
/// `protoc` a mandatory system dependency of every crate that depends on
/// `incin-core`, including the ones that never touch ONNX; a checked-in file
/// costs one regeneration step and removes the dependency entirely.
///
/// Regenerate with `cargo xtask onnx` and verify with `cargo xtask onnx
/// --check`, which is the CI gate that keeps this file equal to what
/// `proto/onnx.proto` compiles to.
#[allow(clippy::doc_overindented_list_items, clippy::enum_variant_names)]
pub mod onnx {
    include!("generated/onnx.rs");
}
