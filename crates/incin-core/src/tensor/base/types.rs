use crate::backend_authoring::{Backend, HostInterop};
use crate::dist::{Local, Placement};
use crate::shapes::{Dyn, Layout};
use crate::shapes::{Shape, ShapeValue};
use crate::tensor::device::{Device, DeviceId};
use crate::tensor::dtype::{ConstDType, DType};
use crate::tensor::grad::{NoGrad, RequiresGrad};
use core::marker::PhantomData;

/// The core `Tensor` type representing an n-dimensional array.
///
/// It holds a reference to a backend-specific tensor representation, while statically tracking
/// its `Shape`, `Backend` (which includes `DType` and `Device`), and its `Grad` requirements.
///
/// `Tensor` is the primary workhorse of the Incin framework. By maintaining shape information
/// directly in the type signature, Incin ensures that tensor operations such as matrix multiplication
/// or convolutions are strictly verified at compile time.
///
/// ## Type Parameters
/// * `S`: The [`Shape`] of the tensor. This can be static (e.g., `s![2, 3, 224, 224]`), dynamic (`Dyn`), or partially dynamic.
/// * `B`: The underlying compute [`Backend`]. It defines how the tensor is stored in memory and how mathematical operations are executed.
/// * `K`: Element [`DType`], which may also be [`crate::shapes::Dyn`] and runtime-checked.
/// * `G`: Trait marker representing whether the tensor requires gradients ([`Grad`](crate::tensor::grad::Grad) or [`NoGrad`]). Defaults to `NoGrad`.
/// * `P`: Logical [`Placement`]. Defaults to [`Local`]; distributed code may
///   select a static placement or [`crate::shapes::Dyn`] for runtime placement metadata.
///
/// ## Examples
///
/// Creating and inspecting statically shaped tensors:
/// ```rust
/// # extern crate incin_core as incin;
/// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
/// use incin::prelude::*;
/// // Compile-time 3D tensor of shape [2, 5, 10]
/// let t = Tensor::<s![2, 5, 10], DefaultBackend>::zeros(()).unwrap();
///
/// assert_eq!(t.dims(), [2, 5, 10]);
/// ```
///
/// Using dynamically shaped tensors:
/// ```rust
/// # extern crate incin_core as incin;
/// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
/// use incin::prelude::*;
/// // Shape determined at runtime
/// let dyn_t = Tensor::<Dyn, DefaultBackend>::ones(vec![32, 64]).unwrap();
///
/// assert_eq!(dyn_t.dims(), vec![32, 64]);
/// ```
pub struct Tensor<
    S: Shape,
    B: Backend,
    K: DType = f32,
    G: RequiresGrad = NoGrad,
    P: Placement = Local,
    // What the type settles about *where* the elements live: strides, offset,
    // alignment, contiguity. Defaulted to `Dyn`, which claims nothing, so
    // every signature written before this parameter existed keeps its meaning.
    L: Layout<S> = Dyn,
> {
    pub(crate) inner: B::Storage<K>,
    pub(crate) _layout: PhantomData<fn() -> L>,
    /// Global logical shape. Backend storage carries this rank's local shape.
    pub(crate) _shape: ShapeValue<S>,
    pub(crate) _dtype: K::Field,
    pub(crate) _device: <B::Device as Device>::Field,
    pub(crate) _grad: G::Field,
    pub(crate) _placement: P::Field,
}

impl<S: Shape, B: Backend, K: DType, G: RequiresGrad, P: Placement, L: Layout<S>> Clone
    for Tensor<S, B, K, G, P, L>
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            _shape: self._shape.clone(),
            _dtype: self._dtype.clone(),
            _device: self._device.clone(),
            _grad: self._grad.clone(),
            _placement: self._placement.clone(),
            _layout: PhantomData,
        }
    }
}

impl<
    S: crate::shapes::Shape,
    B: crate::backend_authoring::Backend + HostInterop,
    K: DType,
    G: RequiresGrad,
    P: Placement,
    L: Layout<S>,
> core::fmt::Display for Tensor<S, B, K, G, P, L>
{
    /// Renders values the way PyTorch's `print(tensor)` does: the backend's
    /// bracketed, right-aligned value grid (`HostInterop::host_format_display`)
    /// wrapped in `tensor(...)`, with nested-bracket rows indented to stay
    /// aligned under the first `[` the way PyTorch's own wrapped output is.
    ///
    /// `dtype=`/`device=`/`requires_grad=` are appended only when they
    /// differ from what a reader would otherwise assume: `f32`
    /// (`DTypeId::default()`), `cpu:0` (`DeviceId::cpu()`), and - not
    /// requiring gradients. That last one is the mirror image of PyTorch's
    /// own rule for the literal reason that `G` defaults to
    /// [`NoGrad`] here rather than `Grad`: printing
    /// `requires_grad=True` whenever `G::requires_grad` is true still means
    /// "printed exactly when true", while the default tensor remains inert.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let prefix = "tensor(";
        let body = B::host_format_display(&self.inner);
        f.write_str(prefix)?;
        for (i, line) in body.lines().enumerate() {
            if i > 0 {
                writeln!(f)?;
                write!(f, "{:1$}", "", prefix.chars().count())?;
            }
            f.write_str(line)?;
        }

        let dtype = self.dtype();
        if dtype != <f32 as ConstDType>::DESCRIPTOR {
            write!(f, ", dtype={}", dtype.name())?;
        }
        if let Ok(device) = <B::Device as Device>::to_incin(&self._device)
            && device != DeviceId::cpu()
        {
            write!(f, ", device={}:{}", device.kind().name(), device.ordinal())?;
        }
        if G::requires_grad(&self._grad) {
            f.write_str(", requires_grad=True")?;
        }
        f.write_str(")")
    }
}

impl<
    S: crate::shapes::Shape,
    B: crate::backend_authoring::Backend + HostInterop,
    K: DType,
    G: RequiresGrad,
    P: Placement,
    L: Layout<S>,
> core::fmt::Debug for Tensor<S, B, K, G, P, L>
{
    /// Prints the backend type name, runtime shape, and the backend's own
    /// debug rendering of its storage.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "Tensor({}, global_shape={:?}, local_shape={:?}, placement={:?}, rank={})\n{}",
            core::any::type_name::<B>(),
            self._shape.shape_buf().as_ref(),
            B::shape(&self.inner).as_ref(),
            self.placement(),
            self.rank_index(),
            B::host_format_debug(&self.inner)
        )
    }
}
