//! Ownership-preserving transfer contracts for tensors and module state.

use crate::err::Result;
use crate::tensor::backend::Backend;
use crate::tensor::device::Device;

/// Transfers an owned value to a new device/backend.
pub trait ToDevice<B: Backend, NewD: Device> {
    /// The transferred value's type.
    type Output;

    /// Moves device-owned storage to `arg`.
    fn to_device(self, arg: &NewD::Arg) -> Result<Self::Output>;
}

impl<T: ToDevice<B, NewD>, B: Backend, NewD: Device> ToDevice<B, NewD> for Option<T> {
    type Output = Option<T::Output>;

    fn to_device(self, arg: &NewD::Arg) -> Result<Self::Output> {
        self.map(|value| value.to_device(arg)).transpose()
    }
}

impl<B: Backend, NewD: Device> ToDevice<B, NewD> for () {
    type Output = ();

    fn to_device(self, _arg: &NewD::Arg) -> Result<Self::Output> {
        Ok(())
    }
}
