use crate::prelude::{DType, Device, RequiresGrad, Shape};

/// Connects a tensor's type parameters to the runtime arguments
/// needed for construction. For each parameter (Shape, DType, Device, Grad),
/// the associated `Arg` type determines what runtime information is needed:
/// - `()` for fully-static parameters (e.g., `Const<N>`, f32, Cpu, Grad)
/// - The actual value for dynamic parameters (e.g., `Vec<usize>`, KindleDType, KindleDevice, bool)
pub trait TensorArgs<S: Shape, K: DType, D: Device, G: RequiresGrad> {
    /// Core abstraction for `Args` within the Kindle framework..
    type Args;
    /// Core abstraction for `construct` within the Kindle framework..
    fn construct(args: Self::Args) -> (S::Field, K::Field, D::Field, G::Field);
}

impl<S, K, D, G> TensorArgs<S, K, D, G> for (S, K, D, G)
where
    S: Shape,
    K: DType,
    D: Device,
    G: RequiresGrad,
{
    /// Core abstraction for `Args` within the Kindle framework..
    type Args = crate::prelude::TensorArgsData<S::Arg, K::Arg, D::Arg, G::Arg>;

    #[inline]
    /// Core abstraction for `construct` within the Kindle framework..
    fn construct(args: Self::Args) -> (S::Field, K::Field, D::Field, G::Field) {
        (
            S::init(args.shape),
            K::init(args.dtype),
            D::init(args.device),
            G::init(args.grad),
        )
    }
}
