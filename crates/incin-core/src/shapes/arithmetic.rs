use typenum::{Diff, Prod, Quot, Sum, U1, U2};

/// Flatten two adjacent dimensions into one.
pub type FlatDim<A, B> = Prod<A, B>;

/// Convolution spatial output dimension on stable Rust.
/// Computed as: (InSize + 2 * Padding - KernelSize) / Stride + 1
pub type ConvOutDim<InSize, KernelSize, Stride, Padding> =
    Sum<Quot<Diff<Sum<InSize, Prod<U2, Padding>>, KernelSize>, Stride>, U1>;
