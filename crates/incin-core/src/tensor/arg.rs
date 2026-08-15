use crate::shapes::{Shape, ShapeBuf};
use crate::tensor::arg_into::TensorArgsData;
use crate::tensor::device::Device;
use crate::tensor::dtype::DType;
use crate::tensor::grad::RequiresGrad;

/// Connects a tensor's type parameters to the runtime arguments
/// needed for construction. For each parameter (Shape, DType, Device, Grad),
/// the associated `Arg` type determines what runtime information is needed:
/// - `()` for fully-static parameters (e.g., `Const<N>`, f32, Cpu, Grad)
/// - The actual value for dynamic parameters (e.g., `Vec<usize>`, DTypeId, DeviceId, bool)
pub trait TensorArgs<S: Shape, K: DType, D: Device, G: RequiresGrad> {
    /// The bundled constructor argument type combining all four parameters' `Arg`s.
    type Args;
    /// Splits the bundled arguments into validated shape storage and fields.
    fn construct(
        args: Self::Args,
    ) -> core::result::Result<
        (ShapeBuf, K::Field, D::Field, G::Field),
        crate::shapes::error::ShapeError,
    >;
}

impl<S, K, D, G> TensorArgs<S, K, D, G> for (S, K, D, G)
where
    S: Shape,
    K: DType,
    D: Device,
    G: RequiresGrad,
{
    /// A struct bundling each parameter's `Arg` (shape dims, dtype id, device id, grad flag).
    type Args = TensorArgsData<S::Arg, K::Arg, D::Arg, G::Arg>;

    #[inline]
    /// Initializes each parameter's `Field` independently from its slot in `args`.
    fn construct(
        args: Self::Args,
    ) -> core::result::Result<
        (ShapeBuf, K::Field, D::Field, G::Field),
        crate::shapes::error::ShapeError,
    > {
        Ok((
            S::resolve(args.shape)?,
            K::init(args.dtype),
            D::init(args.device),
            G::init(args.grad),
        ))
    }
}
