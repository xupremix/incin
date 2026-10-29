/// CIFAR-10/100 (flat binary records, 32x32 RGB).
pub mod cifar;
/// Fashion-MNIST (same IDX layout as MNIST, clothing articles).
pub mod fashion_mnist;
/// Shared IDX archive readers.
pub(crate) mod idx;
/// Mnist.
pub mod mnist;
