use crate::cuda::storage::CudaStorage;
use kindle_core::prelude::*;
use std::fmt::Debug;

/// A dedicated Cuda backend.
#[derive(Debug, Clone, Copy)]
pub struct CudaBackend<T: DType, D: Device> {
    _marker: core::marker::PhantomData<(T, D)>,
}

#[derive(Clone)]
pub struct CudaVar {
    pub storage: CudaStorage,
}

impl<T: DType, D: Device> TensorOps<Self> for CudaBackend<T, D> {
    fn concat<K: DType>(
        tensors: &[&<Self as Backend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        crate::cuda::ops::shape::launch_concat(tensors, dim)
    }
}

impl<T: DType, D: Device> NumericOps<Self> for CudaBackend<T, D> {
    fn add<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        let out =
            crate::cuda::ops::elementwise::launch_binary_op("add", "a + b", lhs, rhs, &out_shape)?;
        let (lhs_shape, rhs_shape) = (lhs.shape.clone(), rhs.shape.clone());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                vec![
                    crate::cuda::tape::unbroadcast(grad_out, &lhs_shape)
                        .expect("unbroadcast lhs (add)"),
                    crate::cuda::tape::unbroadcast(grad_out, &rhs_shape)
                        .expect("unbroadcast rhs (add)"),
                ]
            }),
        });
        Ok(out)
    }

    fn sub<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        let out =
            crate::cuda::ops::elementwise::launch_binary_op("sub", "a - b", lhs, rhs, &out_shape)?;
        let (lhs_shape, rhs_shape) = (lhs.shape.clone(), rhs.shape.clone());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                let neg_grad =
                    crate::cuda::ops::elementwise::launch_unary_op("neg", "-x", grad_out)
                        .expect("neg (sub backward)");
                vec![
                    crate::cuda::tape::unbroadcast(grad_out, &lhs_shape)
                        .expect("unbroadcast lhs (sub)"),
                    crate::cuda::tape::unbroadcast(&neg_grad, &rhs_shape)
                        .expect("unbroadcast rhs (sub)"),
                ]
            }),
        });
        Ok(out)
    }

    fn mul<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        let out =
            crate::cuda::ops::elementwise::launch_binary_op("mul", "a * b", lhs, rhs, &out_shape)?;
        let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
        let (lhs_shape, rhs_shape) = (lhs.shape.clone(), rhs.shape.clone());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                let grad_lhs_shape =
                    crate::cpu::stride::broadcast_shape(&grad_out.shape, &rhs_capture.shape)
                        .expect("mul backward shape (lhs)");
                let grad_lhs = crate::cuda::ops::elementwise::launch_binary_op(
                    "mul",
                    "a * b",
                    grad_out,
                    &rhs_capture,
                    &grad_lhs_shape,
                )
                .expect("mul backward (lhs)");
                let grad_rhs_shape =
                    crate::cpu::stride::broadcast_shape(&grad_out.shape, &lhs_capture.shape)
                        .expect("mul backward shape (rhs)");
                let grad_rhs = crate::cuda::ops::elementwise::launch_binary_op(
                    "mul",
                    "a * b",
                    grad_out,
                    &lhs_capture,
                    &grad_rhs_shape,
                )
                .expect("mul backward (rhs)");
                vec![
                    crate::cuda::tape::unbroadcast(&grad_lhs, &lhs_shape)
                        .expect("unbroadcast lhs (mul)"),
                    crate::cuda::tape::unbroadcast(&grad_rhs, &rhs_shape)
                        .expect("unbroadcast rhs (mul)"),
                ]
            }),
        });
        Ok(out)
    }

    fn div<K: DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let out_shape = crate::cpu::stride::broadcast_shape(&lhs.shape, &rhs.shape)?;
        let out =
            crate::cuda::ops::elementwise::launch_binary_op("div", "a / b", lhs, rhs, &out_shape)?;
        let (lhs_capture, rhs_capture) = (lhs.clone(), rhs.clone());
        let (lhs_shape, rhs_shape) = (lhs.shape.clone(), rhs.shape.clone());
        let (lhs_id, rhs_id, out_id) = (lhs.id, rhs.id, out.id);
        crate::cuda::tape::push(crate::cuda::tape::TapeEntry {
            output_id: out_id,
            input_ids: vec![lhs_id, rhs_id],
            backward: Box::new(move |grad_out: &CudaStorage| {
                // d(lhs/rhs)/dlhs = 1/rhs -> grad_lhs = grad_out / rhs
                let grad_lhs_shape =
                    crate::cpu::stride::broadcast_shape(&grad_out.shape, &rhs_capture.shape)
                        .expect("div backward shape (lhs)");
                let grad_lhs = crate::cuda::ops::elementwise::launch_binary_op(
                    "div",
                    "a / b",
                    grad_out,
                    &rhs_capture,
                    &grad_lhs_shape,
                )
                .expect("div backward (lhs)");
                // d(lhs/rhs)/drhs = -lhs/rhs^2 -> grad_rhs = grad_out * (-lhs/rhs^2)
                let rhs_sq_shape =
                    crate::cpu::stride::broadcast_shape(&rhs_capture.shape, &rhs_capture.shape)
                        .expect("div backward shape (rhs^2)");
                let rhs_sq = crate::cuda::ops::elementwise::launch_binary_op(
                    "mul",
                    "a * b",
                    &rhs_capture,
                    &rhs_capture,
                    &rhs_sq_shape,
                )
                .expect("rhs^2 (div backward)");
                let ratio_shape =
                    crate::cpu::stride::broadcast_shape(&lhs_capture.shape, &rhs_sq.shape)
                        .expect("div backward shape (ratio)");
                let lhs_over_rhs_sq = crate::cuda::ops::elementwise::launch_binary_op(
                    "div",
                    "a / b",
                    &lhs_capture,
                    &rhs_sq,
                    &ratio_shape,
                )
                .expect("lhs/rhs^2 (div backward)");
                let neg_ratio =
                    crate::cuda::ops::elementwise::launch_unary_op("neg", "-x", &lhs_over_rhs_sq)
                        .expect("neg (div backward)");
                let grad_rhs_shape =
                    crate::cpu::stride::broadcast_shape(&grad_out.shape, &neg_ratio.shape)
                        .expect("div backward shape (rhs)");
                let grad_rhs = crate::cuda::ops::elementwise::launch_binary_op(
                    "mul",
                    "a * b",
                    grad_out,
                    &neg_ratio,
                    &grad_rhs_shape,
                )
                .expect("div backward (rhs)");
                vec![
                    crate::cuda::tape::unbroadcast(&grad_lhs, &lhs_shape)
                        .expect("unbroadcast lhs (div)"),
                    crate::cuda::tape::unbroadcast(&grad_rhs, &rhs_shape)
                        .expect("unbroadcast rhs (div)"),
                ]
            }),
        });
        Ok(out)
    }
}

impl<T: DType, D: Device> FloatOps<Self> for CudaBackend<T, D> {}
impl<T: DType, D: Device> CreationOps<Self> for CudaBackend<T, D> {}
impl<T: DType, D: Device> ReductionOps<Self> for CudaBackend<T, D> {}
impl<T: DType, D: Device> QuantizedOps<Self> for CudaBackend<T, D> {}
impl<T: DType, D: Device> OptimizerOps<Self> for CudaBackend<T, D> {}
impl<T: DType, D: Device> ModuleOps<Self> for CudaBackend<T, D> {}
impl<T: DType, D: Device> LossOps<Self> for CudaBackend<T, D> {}

impl<T: DType, D: Device> Backend for CudaBackend<T, D> {
    type Device = D;
    type FloatElem = T;
    type IntElem = f32; // Placeholder

    type Storage<K: DType> = CudaStorage;
    type RawVar = CudaVar;
    type Grads = crate::cuda::tape::CudaGrads;

    type InnerBackend = Self;

    type BackendWithDevice<NewD: Device> = CudaBackend<T, NewD>;

    fn shape<K: DType>(t: &Self::Storage<K>) -> alloc::vec::Vec<usize> {
        t.shape.clone()
    }
    fn format_tensor_display<K: DType>(_t: &Self::Storage<K>) -> alloc::string::String {
        "CudaTensor(...)".to_string()
    }
    fn format_tensor_debug<K: DType>(t: &Self::Storage<K>) -> alloc::string::String {
        format!("CudaTensor(shape={:?})", t.shape)
    }
    fn backward<K: DType>(loss: &Self::Storage<K>) -> Result<Self::Grads> {
        crate::cuda::tape::backward(loss)
    }
    fn backward_with_nan_check<K: DType>(loss: &Self::Storage<K>) -> Result<Self::Grads> {
        crate::cuda::tape::backward_with_nan_check(loss)
    }
    fn get_grad<K: DType>(
        t: &Self::Storage<K>,
        grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>> {
        Ok(grads.grads.get(&t.id).cloned())
    }
    fn to_bytes<K: DType>(_t: &Self::Storage<K>) -> Result<alloc::vec::Vec<u8>> {
        Err(Error::UnsupportedBackendOperation {
            op: "to_bytes",
            backend: "CudaBackend",
        })
    }
    fn from_bytes<K: DType>(
        _bytes: &[u8],
        _shape: &[usize],
        _dtype: KindleDType,
        _device: &KindleDevice,
    ) -> Result<Self::Storage<K>> {
        Err(Error::UnsupportedBackendOperation {
            op: "from_bytes",
            backend: "CudaBackend",
        })
    }
    fn var_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>> {
        Ok(var.storage.clone())
    }
    fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::RawVar> {
        Ok(CudaVar { storage: t.clone() })
    }
    fn var_to_device(var: &Self::RawVar, _device: &KindleDevice) -> Result<Self::RawVar> {
        Ok(var.clone())
    }
    fn assign_var<K: DType>(var: &mut Self::RawVar, tensor: &Self::Storage<K>) -> Result<()> {
        var.storage = tensor.clone();
        Ok(())
    }
}
