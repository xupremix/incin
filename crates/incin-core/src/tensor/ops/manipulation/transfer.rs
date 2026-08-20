//! Device transfer implementation for tensors.

use crate::backend_authoring::{Backend, StorageTransfer, SupportsDType};
use crate::err::Result;
use crate::shapes::Shape;
use crate::tensor::base::Tensor;
use crate::tensor::grad::{NoGrad, RequiresGrad};
use core::marker::PhantomData;

impl<
    S: Shape,
    B: Backend,
    K: crate::tensor::dtype::DType,
    G: RequiresGrad,
    NewD: crate::tensor::device::Device,
> crate::tensor::transfer::ToDevice<B, NewD> for Tensor<S, B, K, G>
where
    B: Backend + StorageTransfer<NewD>,
    <B as StorageTransfer<NewD>>::Output: SupportsDType<K>,
{
    /// The same tensor, rebuilt on backend `NewD`.
    type Output = Tensor<S, <B as StorageTransfer<NewD>>::Output, K, NoGrad>;
    /// Transfers storage to device `arg` and detaches graph tracking.
    fn to_device(self, arg: &NewD::Arg) -> Result<Self::Output> {
        let field = NewD::init(arg.clone());
        let inner = G::grad_mode(&self._grad)
            .restrict(|| B::transfer_storage(&self.inner, &self._dtype, &field))?;
        Tensor::from_shape_value(inner, self._shape, self._dtype, field, PhantomData)
    }
}
