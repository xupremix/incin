use crate::backend_authoring::Backend;
use crate::err::{Error, Result};
use crate::tensor::dtype::DType;
use crate::tensor::grad::RequiresGrad;

/// Shared by every constructor family (`placed`, `local`) that accepts a
/// caller-supplied gradient marker: gradient tracking only makes sense for a
/// floating-point dtype.
pub(super) fn validate_gradient_dtype<B: Backend, K: DType, G: RequiresGrad>(
    dtype: &K::Field,
    grad: &G::Field,
) -> Result<()> {
    if G::requires_grad(grad) && !K::descriptor(dtype).is_float() {
        return Err(Error::UnsupportedDType {
            dtype: K::descriptor(dtype),
            backend: B::BACKEND_NAME,
            op: "gradient tracking",
        });
    }
    Ok(())
}
