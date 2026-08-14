use super::*;


/// Neural-network layer primitives: normalization, embedding lookup,
/// convolution, and pooling. Each takes plain storage (not `Param`/`Module`
/// wrappers) --- the `nn` layer types call through to these.
pub trait ModuleOps<B: Backend> {
    /// Layer normalization over the last dimension: normalizes `t` to zero
    /// mean/unit variance (with `eps` added for numerical stability), then
    /// applies an affine `weight` scale and optional `bias` shift.
    fn layer_norm<K: DType>(
        t: &B::Storage<K>,
        weight: &B::Storage<K>,
        bias: Option<&B::Storage<K>>,
        eps: f32,
    ) -> Result<B::Storage<K>>;
    /// Batch normalization over the channel dimension: normalizes using
    /// batch statistics (training) or `rm`/`rv` running mean/variance
    /// (inference), with `momentum` controlling running-stat updates and
    /// optional affine `w`/`b`.
    fn batch_norm<K: DType>(
        t: &B::Storage<K>,
        w: Option<&B::Storage<K>>,
        b: Option<&B::Storage<K>>,
        rm: Option<&B::Storage<K>>,
        rv: Option<&B::Storage<K>>,
        e: f32,
        momentum: f64,
    ) -> Result<B::Storage<K>>;
    /// Embedding table lookup: gathers rows of the weight matrix `w` at
    /// the integer indices in `t`.
    fn embedding<K: DType, KInt: DType>(
        t: &B::Storage<KInt>,
        w: &B::Storage<K>,
    ) -> Result<B::Storage<K>>;
    /// 1-D convolution of `t` with kernel `w` (and optional bias `b`),
    /// with the given `stride`/`padding`/`dilation`/`groups`.
    fn conv1d<K: DType>(
        t: &B::Storage<K>,
        w: &B::Storage<K>,
        b: Option<&B::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<B::Storage<K>>;
    /// 2-D convolution of `t` with kernel `w` (and optional bias `b`),
    /// with the given `stride`/`padding`/`dilation`/`groups`.
    fn conv2d<K: DType>(
        t: &B::Storage<K>,
        w: &B::Storage<K>,
        b: Option<&B::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<B::Storage<K>>;
    /// Transposed ("deconvolution") 2-D convolution --- the gradient
    /// operation of `conv2d` used as a forward op for upsampling, with an
    /// extra `output_padding` to resolve the otherwise-ambiguous output size.
    fn conv_transpose2d<K: DType>(
        t: &B::Storage<K>,
        w: &B::Storage<K>,
        b: Option<&B::Storage<K>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<B::Storage<K>>;
    /// 2-D max pooling: for each output position, the max over its
    /// `kernel_size` window (given `stride`/`padding`/`dilation`).
    fn max_pool2d<K: DType>(
        t: &B::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<B::Storage<K>>;
    /// 2-D average pooling: for each output position, the mean over its
    /// `kernel_size` window (given `stride`/`padding`).
    fn avg_pool2d<K: DType>(
        t: &B::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<B::Storage<K>>;
    /// Average pooling that derives its own window size per output
    /// position so the output spatial size is exactly `output_size`,
    /// regardless of the input size (PyTorch's `AdaptiveAvgPool2d`).
    fn adaptive_avg_pool2d<K: DType>(
        t: &B::Storage<K>,
        output_size: (usize, usize),
    ) -> Result<B::Storage<K>>;
}

#[path = "dummy.rs"]
mod dummy;
pub use dummy::DummyBackend;
