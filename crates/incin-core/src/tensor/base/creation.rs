use super::Tensor;
use crate::backend_authoring::{Backend, Execute, SupportsDType};
use crate::dist::Local;
use crate::err::{Error, Result};
use crate::exec::Capabilities;
use crate::exec::catalog::{
    ArangeAttributes, CreationAttributes, FullAttributes, LinspaceAttributes, op,
};
use crate::exec::context::ExecutionContext;
use crate::exec::dispatch;
use crate::shapes::{DynShape, Shape, ShapeValue};
use crate::tensor::arg::TensorArgs;
use crate::tensor::arg_into::ArgInto;
use crate::tensor::device::Device;
use crate::tensor::dtype::{BuiltinDType, DType, PlainDType};
use crate::tensor::grad::RequiresGrad;

/// Constructors, which allocate a packed row-major buffer.
///
/// Generic over the layout parameter rather than fixed to the `Unknown`
/// default, so a caller who asks for [`Dense<S, B>`](crate::shapes::Dense) gets
/// a genuine [`RowMajor<S>`](crate::shapes::RowMajor) proof out of the same
/// allocation, while `Tensor<S, B>` keeps meaning exactly what it did.
///
/// The [`FreshDense<S>`](crate::shapes::FreshDense) bound is what keeps that
/// from being a way to forge a layout: it is sealed, and only the layouts a
/// fresh dense allocation actually has implement it. See its documentation for
/// why an unbounded `L` here would become a minting press as soon as a second
/// real layout exists.
impl<S: Shape + DynShape, B: Backend, K: DType, G: RequiresGrad, L> Tensor<S, B, K, G, Local, L>
where
    (S, K, B::Device, G): TensorArgs<S, K, B::Device, G>,
    B: SupportsDType<K>,
    L: crate::shapes::FreshDense<S>,
{
    /// Creates a tensor filled with zeros.
    pub fn zeros<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
        B: Execute<op::Zeros> + Capabilities,
        <B as Execute<op::Zeros>>::Output: Into<B::Storage<K>>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg())?;
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let expected = ShapeValue::<S>::try_new(_shape.clone()).map_err(Error::Shape)?;
        let context = ExecutionContext::from_scope(B::default())
            .with_grad_mode(crate::exec::GradMode::Disabled);
        let inner = dispatch::execute_shaped::<op::Zeros, B, S>(
            &context,
            CreationAttributes {
                shape: _shape.as_ref().to_vec(),
                dtype,
                device,
            },
            &[],
            &expected,
        )?
        .into();
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Creates a tensor filled with ones.
    pub fn ones<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
        B: Execute<op::Ones> + Capabilities,
        <B as Execute<op::Ones>>::Output: Into<B::Storage<K>>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg())?;
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let expected = ShapeValue::<S>::try_new(_shape.clone()).map_err(Error::Shape)?;
        let context = ExecutionContext::from_scope(B::default())
            .with_grad_mode(crate::exec::GradMode::Disabled);
        let inner = dispatch::execute_shaped::<op::Ones, B, S>(
            &context,
            CreationAttributes {
                shape: _shape.as_ref().to_vec(),
                dtype,
                device,
            },
            &[],
            &expected,
        )?
        .into();
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Creates a tensor from a slice whose element type fixes its static dtype.
    ///
    /// Requires a [`PlainDType`]: dtypes with an actual Rust scalar element per
    /// logical value. Block-quantized dtypes (e.g. `Q8_0`) are rejected at
    /// compile time since they have no plain scalar slice representation.
    pub fn from_slice<A>(data: &[<K as PlainDType>::Elem], args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
        K: PlainDType + BuiltinDType,
        B: Execute<op::TensorFromData> + Capabilities,
        <B as Execute<op::TensorFromData>>::Output: Into<B::Storage<K>>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg())?;
        let dims = _shape.clone();
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let byte_len = core::mem::size_of_val(data);
        let bytes = bytemuck::cast_slice(data);
        let expected = ShapeValue::<S>::try_new(_shape.clone()).map_err(Error::Shape)?;
        let context = ExecutionContext::from_scope(B::default())
            .with_grad_mode(crate::exec::GradMode::Disabled);
        let inner = dispatch::execute_shaped_with_payload::<op::TensorFromData, B, S>(
            &context,
            crate::exec::catalog::DataAttributes {
                shape: dims.as_ref().to_vec(),
                dtype,
                device,
                payload: crate::exec::catalog::CreationPayload::Typed { byte_len, dtype },
            },
            &[],
            &expected,
            Some(bytes),
        )?
        .into();
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Creates a tensor from a checked native-endian byte payload.
    pub fn from_bytes<A>(bytes: &[u8], args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
        B: Execute<op::TensorFromBytes> + Capabilities,
        <B as Execute<op::TensorFromBytes>>::Output: Into<B::Storage<K>>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg())?;
        let dims = _shape.clone();
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let expected = ShapeValue::<S>::try_new(_shape.clone()).map_err(Error::Shape)?;
        let context = ExecutionContext::from_scope(B::default())
            .with_grad_mode(crate::exec::GradMode::Disabled);
        let inner = dispatch::execute_shaped_with_payload::<op::TensorFromBytes, B, S>(
            &context,
            crate::exec::catalog::DataAttributes {
                shape: dims.as_ref().to_vec(),
                dtype,
                device,
                payload: crate::exec::catalog::CreationPayload::Bytes {
                    byte_len: bytes.len(),
                },
            },
            &[],
            &expected,
            Some(bytes),
        )?
        .into();
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Creates a tensor filled with random values uniform in [0, 1).
    pub fn rand<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
        B: Execute<op::UniformRandom> + Capabilities,
        <B as Execute<op::UniformRandom>>::Output: Into<B::Storage<K>>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg())?;
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let expected = ShapeValue::<S>::try_new(_shape.clone()).map_err(Error::Shape)?;
        let context = ExecutionContext::from_scope(B::default())
            .with_grad_mode(crate::exec::GradMode::Disabled);
        let inner = dispatch::execute_shaped::<op::UniformRandom, B, S>(
            &context,
            CreationAttributes {
                shape: _shape.as_ref().to_vec(),
                dtype,
                device,
            },
            &[],
            &expected,
        )?
        .into();
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Creates a tensor filled with standard normal random values.
    pub fn randn<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
        B: Execute<op::NormalRandom> + Capabilities,
        <B as Execute<op::NormalRandom>>::Output: Into<B::Storage<K>>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg())?;
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let expected = ShapeValue::<S>::try_new(_shape.clone()).map_err(Error::Shape)?;
        let context = ExecutionContext::from_scope(B::default())
            .with_grad_mode(crate::exec::GradMode::Disabled);
        let inner = dispatch::execute_shaped::<op::NormalRandom, B, S>(
            &context,
            CreationAttributes {
                shape: _shape.as_ref().to_vec(),
                dtype,
                device,
            },
            &[],
            &expected,
        )?
        .into();
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Creates a tensor filled with scalar `val`.
    pub fn full<Sc: Into<crate::tensor::backend::ScalarValue>, A>(val: Sc, args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
        B: Execute<op::Full> + Capabilities,
        <B as Execute<op::Full>>::Output: Into<B::Storage<K>>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg())?;
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let scalar_f64 = val.into().to_f64();
        let expected = ShapeValue::<S>::try_new(_shape.clone()).map_err(Error::Shape)?;
        let context = ExecutionContext::from_scope(B::default())
            .with_grad_mode(crate::exec::GradMode::Disabled);
        let inner = dispatch::execute_shaped::<op::Full, B, S>(
            &context,
            FullAttributes {
                shape: _shape.as_ref().to_vec(),
                dtype,
                device,
                value: scalar_f64,
            },
            &[],
            &expected,
        )?
        .into();
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Creates a 1D tensor starting at `start` with step `step`.
    pub fn arange<Sc: Into<crate::tensor::backend::ScalarValue>, A>(
        start: Sc,
        step: Sc,
        args: A,
    ) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
        B: Execute<op::Arange> + Capabilities,
        <B as Execute<op::Arange>>::Output: Into<B::Storage<K>>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg())?;
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let s_f64 = start.into().to_f64();
        let st_f64 = step.into().to_f64();
        let expected = ShapeValue::<S>::try_new(_shape.clone()).map_err(Error::Shape)?;
        let context = ExecutionContext::from_scope(B::default())
            .with_grad_mode(crate::exec::GradMode::Disabled);
        let inner = dispatch::execute_shaped::<op::Arange, B, S>(
            &context,
            ArangeAttributes {
                shape: _shape.as_ref().to_vec(),
                dtype,
                device,
                start: s_f64,
                step: st_f64,
            },
            &[],
            &expected,
        )?
        .into();
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Creates a 1D tensor with linearly spaced values between `start` and `end`.
    pub fn linspace<Sc: Into<crate::tensor::backend::ScalarValue>, A>(
        start: Sc,
        end: Sc,
        args: A,
    ) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
        B: Execute<op::Linspace> + Capabilities,
        <B as Execute<op::Linspace>>::Output: Into<B::Storage<K>>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg())?;
        let device = B::Device::to_incin(&_device)?;
        let dtype = B::resolve_dtype(&_dtype, &device)?;
        let s_f64 = start.into().to_f64();
        let e_f64 = end.into().to_f64();
        let expected = ShapeValue::<S>::try_new(_shape.clone()).map_err(Error::Shape)?;
        let context = ExecutionContext::from_scope(B::default())
            .with_grad_mode(crate::exec::GradMode::Disabled);
        let inner = dispatch::execute_shaped::<op::Linspace, B, S>(
            &context,
            LinspaceAttributes {
                shape: _shape.as_ref().to_vec(),
                dtype,
                device,
                start: s_f64,
                end: e_f64,
            },
            &[],
            &expected,
        )?
        .into();
        Self::from_parts(inner, _shape, _dtype, _device, _grad)
    }

    /// Wraps an existing backend storage in a Tensor.
    pub fn from_raw<A>(raw_tensor: B::Storage<K>, args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, G) as TensorArgs<S, K, B::Device, G>>::Args>,
    {
        let (_shape, _dtype, _device, _grad) = <(S, K, B::Device, G)>::construct(args.into_arg())?;
        Self::from_parts(raw_tensor, _shape, _dtype, _device, _grad)
    }
}
