use cudarc::nvrtc::compile_ptx;
fn main() {
    let ptx = compile_ptx("extern \"C\" __global__ void kernel() {}").unwrap();
    let bytes = ptx.image;
}
