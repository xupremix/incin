use crate::prelude::{DType, Device, RequiresGrad, Shape};

/// Connects a tensor's type parameters to the runtime arguments
/// needed for construction. For each parameter (Shape, DType, Device, Grad),
/// the associated `Arg` type determines what runtime information is needed:
/// - `()` for fully-static parameters (e.g., `Const<N>`, f32, Cpu, Grad)
/// - The actual value for dynamic parameters (e.g., `Vec<usize>`, KindleDType, KindleDevice, bool)
pub trait TensorArgs<S: Shape, T: DType, D: Device, G: RequiresGrad> {
    type Args;
    fn construct(args: Self::Args) -> (S::Field, T::Field, D::Field, G::Field);
}

impl<S, T, D, G> TensorArgs<S, T, D, G> for (S, T, D, G)
where
    S: Shape,
    T: DType,
    D: Device,
    G: RequiresGrad,
{
    type Args = (S::Arg, T::Arg, D::Arg, G::Arg);

    #[inline]
    fn construct(args: Self::Args) -> (S::Field, T::Field, D::Field, G::Field) {
        (
            S::init(args.0),
            T::init(args.1),
            D::init(args.2),
            G::init(args.3),
        )
    }
}
