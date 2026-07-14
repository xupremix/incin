#[cfg(feature = "cuda")]
use cudarc::driver::CudaContext;
fn main() {
    #[cfg(feature = "cuda")]
    {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let _ = stream.alloc_zeros::<u8>(100).unwrap();
    }
}
