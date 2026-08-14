use super::*;

pub trait FloatOps<B: Backend> {
    /// Rectified linear unit: `max(0, x)`.
    fn relu<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Heaviside step function: `1` where `x > 0`, else `0`.
    fn step<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Mish activation: `x * tanh(softplus(x))`.
    fn mish<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Exponential Linear Unit: `x` where `x > 0`, else `exp(x) - 1`.
    fn elu<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Gaussian Error Linear Unit (exact, erf-based):
    /// `x * 0.5 * (1 + erf(x / sqrt(2)))`.
    fn gelu<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise absolute value.
    fn abs<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise natural exponential.
    fn exp<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise negation: `-x`.
    fn neg<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise square root.
    fn sqrt<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise natural logarithm.
    fn log<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise hyperbolic tangent.
    fn tanh<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise logistic sigmoid: `1 / (1 + exp(-x))`.
    fn sigmoid<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Swish/SiLU activation: `x * sigmoid(x)`.
    fn swish<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Softmax along `dim`, numerically stabilized by subtracting the
    /// per-slice max before exponentiating.
    fn softmax<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// Adds scalar `scalar` to every element.
    fn add_scalar_float<K: DType>(_t: &B::Storage<K>, _scalar: f64) -> Result<B::Storage<K>>;
    /// Multiplies every element by scalar `scalar`.
    fn mul_scalar_float<K: DType>(_t: &B::Storage<K>, _scalar: f64) -> Result<B::Storage<K>>;
    /// Elementwise power by float exponent `exponent`.
    fn powf<K: DType>(_t: &B::Storage<K>, _exponent: f64) -> Result<B::Storage<K>>;
    /// Elementwise clamp to `[min, max]`.
    fn clamp<K: DType>(_t: &B::Storage<K>, _min: f64, _max: f64) -> Result<B::Storage<K>>;
    /// Elementwise sign (-1.0 for negative, 0.0 for zero, +1.0 for positive).
    fn sign<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise floor.
    fn floor<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise ceil.
    fn ceil<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise round.
    fn round<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise base-2 logarithm.
    fn log2<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise base-10 logarithm.
    fn log10<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise sine.
    fn sin<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise cosine.
    fn cos<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise tangent.
    fn tan<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise arcsine.
    fn asin<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise arccosine.
    fn acos<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise arctangent.
    fn atan<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise 2-argument arctangent.
    fn atan2<K: DType>(_y: &B::Storage<K>, _x: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise hyperbolic sine.
    fn sinh<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise hyperbolic cosine.
    fn cosh<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise inverse hyperbolic sine.
    fn asinh<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise inverse hyperbolic cosine.
    fn acosh<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise inverse hyperbolic tangent.
    fn atanh<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise error function.
    fn erf<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise reciprocal square root: 1 / sqrt(x).
    fn rsqrt<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise truncation toward zero.
    fn trunc<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise fractional part: x - trunc(x).
    fn frac<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise floating point remainder `x % y`.
    fn fmod<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Elementwise IEEE remainder.
    fn remainder<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
}

/// Shape, layout, and dtype manipulation that doesn't change element
/// values (aside from `tensor_to_dtype`'s cast) --- reshapes, views,
/// concatenation, and host-readback conversions.
pub trait TensorOps<B: Backend> {
    /// Reinterprets storage under a new `shape` with the same element count
    /// and row-major ordering (no data movement on backends with
    /// contiguous storage).
    fn reshape<K: DType>(_t: &B::Storage<K>, _shape: &[usize]) -> Result<B::Storage<K>>;
    /// Swaps dimensions `dim1` and `dim2` in the logical shape (a view, not
    /// a copy, on backends with strided storage).
    fn transpose<K: DType>(_t: &B::Storage<K>, _dim1: usize, _dim2: usize)
    -> Result<B::Storage<K>>;
    /// Batched matrix multiplication over the trailing two dimensions of
    /// `lhs`/`rhs`, broadcasting any leading batch dimensions.
    fn matmul<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Broadcasts `t` to `shape` per NumPy rules (each dimension where the
    /// source size differs from the target must be exactly 1).
    fn broadcast_as<K: DType>(_t: &B::Storage<K>, _shape: &[usize]) -> Result<B::Storage<K>>;
    /// Takes the `len`-element window `[start, start + len)` along `dim`,
    /// keeping every other dimension unchanged.
    fn narrow<K: DType>(
        _t: &B::Storage<K>,
        _dim: usize,
        _start: usize,
        _len: usize,
    ) -> Result<B::Storage<K>>;
    /// Removes dimension `dim`, which must have size 1.
    fn squeeze<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// Stacks same-shaped tensors along a brand-new dimension inserted at
    /// `dim` (output has one more dimension than each input).
    fn stack<K: DType>(_t: &[&B::Storage<K>], _dim: usize) -> Result<B::Storage<K>>;
    /// Concatenates tensors along an existing dimension `dim` (every other
    /// dimension must already match across inputs).
    fn concat<K: DType>(_t: &[&B::Storage<K>], _dim: usize) -> Result<B::Storage<K>>;
    /// Takes a `[start, end)` window per dimension, one `(start, end)` pair
    /// in `ranges` for each dimension of `t`, in order.
    fn slice<K: DType>(_t: &B::Storage<K>, _ranges: &[(usize, usize)]) -> Result<B::Storage<K>>;
    /// Collapses dimensions `[start_dim, end_dim]` (inclusive) into a
    /// single dimension, preserving element order.
    fn flatten<K: DType>(
        _t: &B::Storage<K>,
        _start_dim: usize,
        _end_dim: usize,
    ) -> Result<B::Storage<K>>;
    /// Selects elements from `on_true` where `mask` is true, and `on_false` elsewhere.
    fn where_cond<K: DType>(
        _mask: &B::Storage<bool>,
        _on_true: &B::Storage<K>,
        _on_false: &B::Storage<K>,
    ) -> Result<B::Storage<K>>;
    /// Gathers values along `dim` using `index` tensor.
    fn gather<K: DType, KInt: DType>(
        _t: &B::Storage<K>,
        _dim: usize,
        _index: &B::Storage<KInt>,
    ) -> Result<B::Storage<K>>;
    /// Scatters `src` values along `dim` into `t` using `index` tensor.
    fn scatter<K: DType, KInt: DType>(
        _t: &B::Storage<K>,
        _dim: usize,
        _index: &B::Storage<KInt>,
        _src: &B::Storage<K>,
    ) -> Result<B::Storage<K>>;
    /// Selects slice along `dim` according to 1D `index` tensor.
    fn index_select<K: DType, KInt: DType>(
        _t: &B::Storage<K>,
        _dim: usize,
        _index: &B::Storage<KInt>,
    ) -> Result<B::Storage<K>>;
    /// Fills elements of `t` where `mask` is true with scalar `value`.
    fn masked_fill<K: DType>(
        _t: &B::Storage<K>,
        _mask: &B::Storage<bool>,
        _value: f64,
    ) -> Result<B::Storage<K>>;
    /// Inserts a 1-sized dimension at `dim`.
    fn unsqueeze<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// Repeats tensor data along each dimension according to `repeats`.
    fn repeat<K: DType>(_t: &B::Storage<K>, _repeats: &[usize]) -> Result<B::Storage<K>>;
    /// Pads tensor with `val` according to `padding` (before, after) pairs.
    fn pad<K: DType>(
        _t: &B::Storage<K>,
        _padding: &[(usize, usize)],
        _val: f64,
    ) -> Result<B::Storage<K>>;
    /// Retains upper triangular part of matrix, zeroing the rest.
    fn triu<K: DType>(_t: &B::Storage<K>, _k: i64) -> Result<B::Storage<K>>;
    /// Retains lower triangular part of matrix, zeroing the rest.
    fn tril<K: DType>(_t: &B::Storage<K>, _k: i64) -> Result<B::Storage<K>>;
    /// Extracts diagonal or constructs diagonal matrix.
    fn diag<K: DType>(_t: &B::Storage<K>, _k: i64) -> Result<B::Storage<K>>;
    /// Reads a single-element floating-point tensor's value as `f64`.
    /// Errors if `t` has more than one element.
    fn float_to_scalar<K: DType>(_t: &B::Storage<K>) -> Result<f64>;

    /// Element-wise equality (`self == rhs`).
    fn cmp_eq<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<bool>>;
    /// Element-wise inequality (`self != rhs`).
    fn cmp_ne<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<bool>>;
    /// Element-wise less-than (`self < rhs`).
    fn cmp_lt<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<bool>>;
    /// Element-wise less-than-or-equal (`self <= rhs`).
    fn cmp_le<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<bool>>;
    /// Element-wise greater-than (`self > rhs`).
    fn cmp_gt<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<bool>>;
    /// Element-wise greater-than-or-equal (`self >= rhs`).
    fn cmp_ge<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<bool>>;

    /// Logical AND.
    fn logical_and(_lhs: &B::Storage<bool>, _rhs: &B::Storage<bool>) -> Result<B::Storage<bool>>;
    /// Logical OR.
    fn logical_or(_lhs: &B::Storage<bool>, _rhs: &B::Storage<bool>) -> Result<B::Storage<bool>>;
    /// Logical NOT.
    fn logical_not(_t: &B::Storage<bool>) -> Result<B::Storage<bool>>;

    /// Subtract scalar (`self - scalar`).
    fn sub_scalar<K: DType>(_t: &B::Storage<K>, _val: f64) -> Result<B::Storage<K>>;
    /// Divide scalar (`self / scalar`).
    fn div_scalar<K: DType>(_t: &B::Storage<K>, _val: f64) -> Result<B::Storage<K>>;

    /// Element-wise maximum of two tensors.
    fn maximum<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Element-wise minimum of two tensors.
    fn minimum<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Element-wise absolute difference `|lhs - rhs|`.
    fn abs_diff<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Linear interpolation `start + weight * (end - start)`.
    fn lerp<K: DType>(
        _start: &B::Storage<K>,
        _end: &B::Storage<K>,
        _weight: f64,
    ) -> Result<B::Storage<K>>;

    /// Fused add-matmul: `beta * mat + alpha * (mat1 x mat2)`.
    fn addmm<K: DType>(
        _mat: &B::Storage<K>,
        _mat1: &B::Storage<K>,
        _mat2: &B::Storage<K>,
        _beta: f64,
        _alpha: f64,
    ) -> Result<B::Storage<K>>;
    /// Batched matrix multiplication for 3D tensors.
    fn bmm<K: DType>(_lhs: &B::Storage<K>, _rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Scaled Dot-Product Attention: `softmax(q * k^T / scale) * v`.
    fn scaled_dot_product_attention<K: DType>(
        _q: &B::Storage<K>,
        _k: &B::Storage<K>,
        _v: &B::Storage<K>,
        _mask: Option<&B::Storage<K>>,
        _scale: Option<f64>,
    ) -> Result<B::Storage<K>>;

    /// Sliding window extraction along `dim`.
    fn unfold<K: DType>(
        _t: &B::Storage<K>,
        _dim: usize,
        _size: usize,
        _step: usize,
    ) -> Result<B::Storage<K>>;
    /// Pixel shuffle upscaling for 4D (N, C, H, W) tensors.
    fn pixel_shuffle<K: DType>(_t: &B::Storage<K>, _upscale_factor: usize)
    -> Result<B::Storage<K>>;
    /// Group normalization across `groups`.
    fn group_norm<K: DType>(_t: &B::Storage<K>, _groups: usize, _eps: f64)
    -> Result<B::Storage<K>>;
    /// Instance normalization for 4D (N, C, H, W) tensors.
    fn instance_norm<K: DType>(_t: &B::Storage<K>, _eps: f64) -> Result<B::Storage<K>>;

    /// Prepends size-1 dimensions on the left until `t` has as many
    /// dimensions as `shape`, then broadcasts to `shape` (the NumPy
    /// "align on the right" convention for broadcasting mismatched ranks).
    fn broadcast_left<K: DType>(_t: &B::Storage<K>, _shape: &[usize]) -> Result<B::Storage<K>>;
    /// Reads a 1-D floating-point tensor's values into a host `Vec<f64>`.
    fn float_to_vec1<K: DType>(_t: &B::Storage<K>) -> Result<alloc::vec::Vec<f64>>;

    /// Reads a single-element integer tensor's value as `i64`. Errors if
    /// `t` has more than one element.
    fn int_to_scalar<K: DType>(_t: &B::Storage<K>) -> Result<i64>;
    /// Reads a 1-D integer tensor's values into a host `Vec<i64>`.
    fn int_to_vec1<K: DType>(_t: &B::Storage<K>) -> Result<alloc::vec::Vec<i64>>;

    /// Casts storage from dtype `K` to dtype `K2`, converting element
    /// values (not a bit-reinterpret --- see `dtype` for the target's
    /// `DTypeId`).
    fn tensor_to_dtype<K: DType, K2: DType>(
        _t: &B::Storage<K>,
        _dtype: DTypeDescriptor,
    ) -> Result<B::Storage<K2>>;
}


/// Reductions that collapse a tensor along one or all dimensions ---
/// aggregate statistics (`sum`/`mean`/`max`/`min`) and index-producing
/// selections (`argmax`/`argmin`/`topk`/`argsort`).
pub trait ReductionOps<B: Backend> {
    /// Sums every element into a single-element tensor.
    fn sum_all<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Averages every element into a single-element tensor.
    fn mean_all<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Reduces to the single largest element.
    fn max_all<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Reduces to the single smallest element.
    fn min_all<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Sums along `dim`, removing that dimension from the output shape.
    fn sum_dim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// Sums along `dim`, keeping it in the output shape as size 1.
    fn sum_keepdim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// Averages along `dim`, removing that dimension from the output shape.
    fn mean_dim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// Averages along `dim`, keeping it in the output shape as size 1.
    fn mean_keepdim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// Reduces along `dim` to its max, removing that dimension.
    fn max_dim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// Reduces along `dim` to its max, keeping it in the output shape as
    /// size 1.
    fn max_keepdim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// Reduces along `dim` to its min, removing that dimension.
    fn min_dim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// Reduces along `dim` to its min, keeping it in the output shape as
    /// size 1.
    fn min_keepdim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// Index of the maximum element, either flattened (`dim: None`) or
    /// along a single `dim`.
    fn argmax<K: DType, KInt: DType>(
        t: &B::Storage<K>,
        dim: Option<usize>,
    ) -> Result<B::Storage<KInt>>;
    /// Index of the minimum element, either flattened (`dim: None`) or
    /// along a single `dim`.
    fn argmin<K: DType, KInt: DType>(
        t: &B::Storage<K>,
        dim: Option<usize>,
    ) -> Result<B::Storage<KInt>>;
    /// Product of all elements in tensor.
    fn prod_all<K: DType>(_t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Product of elements along `dim`.
    fn prod_dim<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// Cumulative sum along `dim`.
    fn cumsum<K: DType>(_t: &B::Storage<K>, _dim: usize) -> Result<B::Storage<K>>;
    /// The `k` largest (`largest: true`) or smallest (`largest: false`)
    /// elements along `dim`, returned as `(values, indices)`.
    fn topk<K: DType, KInt: DType>(
        t: &B::Storage<K>,
        k: usize,
        dim: usize,
        largest: bool,
    ) -> Result<(B::Storage<K>, B::Storage<KInt>)>;
    /// Indices that would sort `t` along `dim`, ascending or `descending`.
    fn argsort<K: DType, KInt: DType>(
        t: &B::Storage<K>,
        dim: usize,
        descending: bool,
    ) -> Result<B::Storage<KInt>>;
}

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
