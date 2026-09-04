//! Host interop, dtype casting, and scalar/vector data extraction.

use crate::backend_authoring::Execute;
use crate::backend_authoring::{Backend, HostInterop};
use crate::dist::placement::Local;
use crate::err::Result;
use crate::exec::Capabilities;
use crate::exec::catalog::{DTypeAttributes, op};
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::shapes::Layout;
use crate::shapes::{DynShape, Shape};
use crate::tensor::base::Tensor;
use crate::tensor::grad::RequiresGrad;

pub(crate) fn is_valid_scalar_type<E: 'static>() -> bool {
    let tid = core::any::TypeId::of::<E>();
    tid == core::any::TypeId::of::<bool>()
        || tid == core::any::TypeId::of::<u8>()
        || tid == core::any::TypeId::of::<u16>()
        || tid == core::any::TypeId::of::<u32>()
        || tid == core::any::TypeId::of::<u64>()
        || tid == core::any::TypeId::of::<usize>()
        || tid == core::any::TypeId::of::<i8>()
        || tid == core::any::TypeId::of::<i16>()
        || tid == core::any::TypeId::of::<i32>()
        || tid == core::any::TypeId::of::<i64>()
        || tid == core::any::TypeId::of::<isize>()
        || tid == core::any::TypeId::of::<f32>()
        || tid == core::any::TypeId::of::<f64>()
        || tid == core::any::TypeId::of::<half::f16>()
        || tid == core::any::TypeId::of::<half::bf16>()
}

/// Whether `E` is the exact Rust type the tensor's `dtype` stores.
///
/// The extraction below reads the tensor's bytes through a `*const E`, so
/// agreeing on a byte *width* is not enough: `f32` and `u32` are both four
/// bytes wide, and reading one as the other reinterprets the bit pattern
/// instead of converting it. `1.0f32` extracted as `u32` returned
/// `1065353216` rather than reporting a mismatch, which is a wrong answer
/// with no error attached to it.
///
/// `bool` is deliberately absent. It is not a stored dtype at all; both
/// callers handle it before reaching here, as a per-element truthy test
/// rather than a reinterpret.
///
/// `Q8_0` is also absent, and matches nothing: a block-quantized element has
/// no scalar Rust type to be read as without dequantizing first.
pub(crate) fn scalar_type_matches_dtype<E: 'static>(
    dtype: crate::tensor::dtype::DTypeDescriptor,
) -> bool {
    use crate::tensor::dtype::DTypeId;
    let tid = core::any::TypeId::of::<E>();
    match dtype.builtin_id() {
        Some(DTypeId::U8) => tid == core::any::TypeId::of::<u8>(),
        Some(DTypeId::U32) => tid == core::any::TypeId::of::<u32>(),
        Some(DTypeId::I64) => tid == core::any::TypeId::of::<i64>(),
        Some(DTypeId::BF16) => tid == core::any::TypeId::of::<half::bf16>(),
        Some(DTypeId::F16) => tid == core::any::TypeId::of::<half::f16>(),
        Some(DTypeId::F32) => tid == core::any::TypeId::of::<f32>(),
        Some(DTypeId::F64) => tid == core::any::TypeId::of::<f64>(),
        Some(DTypeId::Bool) => tid == core::any::TypeId::of::<bool>(),
        _ => false,
    }
}

impl<S: Shape + DynShape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad, L: Layout>
    Tensor<S, B, K, G, Local, L>
{
    /// Cast the tensor's elements to another dtype.
    pub fn to_dtype<T2: crate::tensor::dtype::DType<Arg = ()>>(
        &self,
    ) -> Result<crate::shapes::Dense<S, B, T2, G, Local>>
    where
        B: Execute<op::ToDType> + Capabilities,
        <B as Execute<op::ToDType>>::Output: Into<B::Storage<T2>>,
    {
        let field = T2::init(());
        let descriptor = T2::descriptor(&field);
        let input = TensorHandle::from_storage::<B, K, Local>(&self.inner);
        let context = crate::tensor::grad::execution_context::<B, G>(&self._grad);
        let inner = G::grad_mode(&self._grad)
            .restrict(|| {
                dispatch::execute_shaped::<op::ToDType, B, S>(
                    &context,
                    DTypeAttributes { dtype: descriptor },
                    &[input],
                    &self._shape,
                )
            })?
            .into();
        Tensor::from_shape_value(
            inner,
            self._shape.clone(),
            field,
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Extracts a single scalar value from a 0D or 1D tensor.
    /// This will bring the tensor data to the CPU and read the bytes.
    ///
    /// `bool` is handled as a truthy (any-nonzero-byte) conversion rather
    /// than a raw reinterpret, regardless of whether the tensor's actual
    /// dtype element size happens to match `size_of::<bool>()`: `bool` has
    /// only two valid bit patterns (`0x00`/`0x01`), and there is no
    /// `DTypeId::Bool` (ONNX-style boolean tensors are stored as another
    /// dtype, typically `U8`, and read out via this truthy conversion), so
    /// reinterpreting an arbitrary stored byte as `bool` via
    /// `read_unaligned` would be undefined behavior whenever that byte
    /// isn't `0` or `1`.
    pub fn to_scalar<E: Copy + 'static>(&self) -> Result<E>
    where
        B: HostInterop,
    {
        if !is_valid_scalar_type::<E>() {
            return Err(crate::err::Error::Msg(alloc::format!(
                "Invalid target scalar type for tensor extraction: {:?}",
                core::any::type_name::<E>()
            )));
        }

        let num_elements = S::checked_numel(
            &self.shape_buf_value(),
            crate::shapes::error::OperationKind::Storage,
        )?;
        if num_elements != 1 {
            return Err(crate::err::Error::Msg(alloc::format!(
                "to_scalar requires a tensor with exactly 1 element, but tensor has {num_elements} elements"
            )));
        }

        let bytes = B::to_bytes(&self.inner)?;
        let dtype = self.dtype();

        if !scalar_type_matches_dtype::<E>(dtype) {
            return Err(crate::err::Error::Msg(alloc::format!(
                "Type mismatch when converting to scalar. Tensor dtype {:?} cannot be extracted as {}: the bytes would be reinterpreted rather than converted",
                dtype,
                core::any::type_name::<E>()
            )));
        }

        if core::any::TypeId::of::<E>() == core::any::TypeId::of::<bool>() {
            if bytes.len() != 1 {
                return Err(crate::err::Error::Msg(
                    "cannot convert an empty tensor to a bool scalar".into(),
                ));
            }
            let byte = bytes[0];
            let val = match byte {
                0 => false,
                1 => true,
                other => {
                    return Err(crate::err::Error::Msg(alloc::format!(
                        "Invalid boolean storage byte: expected 0 or 1, found {}",
                        other
                    )));
                }
            };
            // SAFETY: `E` is verified to be exactly `bool` above.
            return Ok(unsafe { core::ptr::read_unaligned(&val as *const bool as *const E) });
        }

        let elem_size = core::mem::size_of::<E>();
        let expected_size = dtype.encoding().scalar_bytes().unwrap_or(0);
        if bytes.len() != elem_size || elem_size != expected_size {
            return Err(crate::err::Error::Msg(alloc::format!(
                "Size mismatch when converting to scalar. Tensor dtype {:?} ({} bytes) vs requested type ({} bytes)",
                dtype,
                bytes.len(),
                elem_size
            )));
        }
        // SAFETY: `E` is verified to be a primitive scalar numeric type and bytes.len() == elem_size.
        let val = unsafe { core::ptr::read_unaligned(bytes.as_ptr() as *const E) };
        Ok(val)
    }

    /// Extracts a 1D vector of scalars from this tensor.
    pub fn to_vec1<E: Copy + 'static>(&self) -> Result<alloc::vec::Vec<E>>
    where
        B: HostInterop,
    {
        if !is_valid_scalar_type::<E>() {
            return Err(crate::err::Error::Msg(alloc::format!(
                "Invalid target scalar type for vector extraction: {:?}",
                core::any::type_name::<E>()
            )));
        }

        let bytes = B::to_bytes(&self.inner)?;
        let num_elements = S::checked_numel(
            &self.shape_buf_value(),
            crate::shapes::error::OperationKind::Storage,
        )?;
        let dtype = self.dtype();

        if !scalar_type_matches_dtype::<E>(dtype) {
            return Err(crate::err::Error::Msg(alloc::format!(
                "Type mismatch when converting to vec. Tensor dtype {:?} cannot be extracted as {}: the bytes would be reinterpreted rather than converted",
                dtype,
                core::any::type_name::<E>()
            )));
        }

        if core::any::TypeId::of::<E>() == core::any::TypeId::of::<bool>() {
            if bytes.len() != num_elements {
                return Err(crate::err::Error::Msg(alloc::format!(
                    "Size mismatch when converting to vec. Tensor dtype bytes: {}, expected: {}",
                    bytes.len(),
                    num_elements
                )));
            }
            let mut out = alloc::vec::Vec::with_capacity(num_elements);
            for &byte in &bytes {
                let val = match byte {
                    0 => false,
                    1 => true,
                    other => {
                        return Err(crate::err::Error::Msg(alloc::format!(
                            "Invalid boolean storage byte: expected 0 or 1, found {}",
                            other
                        )));
                    }
                };
                // SAFETY: `E` is verified to be exactly `bool` above.
                out.push(unsafe { core::ptr::read_unaligned(&val as *const bool as *const E) });
            }
            return Ok(out);
        }

        if !scalar_type_matches_dtype::<E>(dtype) {
            return Err(crate::err::Error::Msg(alloc::format!(
                "Type mismatch when converting to vec. Tensor dtype {:?} cannot be extracted as {}: the bytes would be reinterpreted rather than converted",
                dtype,
                core::any::type_name::<E>()
            )));
        }

        let elem_size = core::mem::size_of::<E>();
        let expected_elem_size = dtype.encoding().scalar_bytes().unwrap_or(0);
        if elem_size != expected_elem_size {
            return Err(crate::err::Error::Msg(alloc::format!(
                "Element size mismatch converting to vec: Tensor dtype {:?} element size {} vs requested type size {}",
                dtype,
                expected_elem_size,
                elem_size
            )));
        }
        let expected_bytes = num_elements * elem_size;
        if bytes.len() != expected_bytes {
            return Err(crate::err::Error::Msg(alloc::format!(
                "Size mismatch when converting to vec. Tensor dtype bytes: {}, expected: {}",
                bytes.len(),
                expected_bytes
            )));
        }
        let mut out = alloc::vec::Vec::with_capacity(num_elements);
        for chunk in bytes.chunks_exact(elem_size) {
            // SAFETY: `E` is verified to be a primitive scalar type above and chunk is elem_size bytes.
            let val = unsafe { core::ptr::read_unaligned(chunk.as_ptr() as *const E) };
            out.push(val);
        }
        Ok(out)
    }
}
