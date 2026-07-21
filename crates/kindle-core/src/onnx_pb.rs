/// Core abstraction for `onnx` within the Kindle framework..
#[allow(clippy::doc_overindented_list_items, clippy::enum_variant_names)]
pub mod onnx {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}
