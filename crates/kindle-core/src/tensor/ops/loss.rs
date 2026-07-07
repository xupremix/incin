//! Common loss functions (MSE, L1, BCE, CrossEntropy) for training.
//!
//! This module provides standard loss functions used to train neural networks.
//! Loss functions automatically compute and track their required reduction shape
//! (e.g. reducing down to a scalar or maintaining a batched shape) using type-level
//! logic to ensure that backpropagation can flow correctly from the scalar loss.
use crate::prelude::{Backend, Dyn, DynShape, RequiresGrad, Result, Shape, Tensor};

impl<S: Shape, B: Backend, G: RequiresGrad> Tensor<S, B, G> {
    /// Dynamically concatenates a slice of tensors along `dim`.
    /// This is fallible at runtime if shapes mismatch or dim is out of bounds.
    pub fn try_concat_slice(tensors: &[&Tensor<S, B, G>], dim: usize) -> Result<Tensor<Dyn, B, G>> {
        let raw_tensors: alloc::vec::Vec<&B::RawTensor> =
            tensors.iter().map(|t| &t.inner).collect();
        if raw_tensors.is_empty() {
            return Err(crate::err::Error::Msg(
                "Cannot concat empty list".to_string(),
            ));
        }
        let inner = B::concat(&raw_tensors, dim)?;
        let mut out_shape = B::shape(&tensors[0].inner);
        out_shape[dim] = tensors.iter().map(|t| B::shape(&t.inner)[dim]).sum();
        Ok(Tensor::from_parts_unchecked(
            inner,
            <Dyn as Shape>::from_dyn(&out_shape).unwrap(),
            tensors[0]._dtype.clone(),
            tensors[0]._device.clone(),
            tensors[0]._grad.clone(),
        ))
    }

    /// Statically concatenates `self` with `other` along `Axis`.
    pub fn concat<S2, Axis>(
        &self,
        other: &Tensor<S2, B, G>,
    ) -> Result<Tensor<<S as crate::shapes::concat::ConcatShape<S2, Axis>>::Output, B, G>>
    where
        S2: Shape,
        Axis: typenum::Unsigned,
        S: crate::shapes::concat::ConcatShape<S2, Axis>,
        <<S as crate::shapes::concat::ConcatShape<S2, Axis>>::Output as Shape>::Field:
            core::default::Default,
    {
        let dim = Axis::USIZE;
        let inner = B::concat(&[&self.inner, &other.inner], dim)?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            core::default::Default::default(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Dynamically concatenates `self` with `other` along `dim`.
    pub fn try_concat<S2>(&self, other: &Tensor<S2, B, G>, dim: usize) -> Result<Tensor<Dyn, B, G>>
    where
        S2: Shape,
    {
        let inner = B::concat(&[&self.inner, &other.inner], dim)?;
        let mut out_shape = B::shape(&self.inner);
        out_shape[dim] += B::shape(&other.inner)[dim];
        Ok(Tensor::from_parts_unchecked(
            inner,
            <Dyn as Shape>::from_dyn(&out_shape).unwrap(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Dynamically stacks a slice of tensors along `dim`.
    pub fn try_stack_slice(tensors: &[&Tensor<S, B, G>], dim: usize) -> Result<Tensor<Dyn, B, G>> {
        let raw_tensors: alloc::vec::Vec<&B::RawTensor> =
            tensors.iter().map(|t| &t.inner).collect();
        if raw_tensors.is_empty() {
            return Err(crate::err::Error::Msg(
                "Cannot stack empty list".to_string(),
            ));
        }
        let inner = B::stack(&raw_tensors, dim)?;
        let mut out_shape = B::shape(&tensors[0].inner);
        out_shape.insert(dim, tensors.len());
        Ok(Tensor::from_parts_unchecked(
            inner,
            <Dyn as Shape>::from_dyn(&out_shape).unwrap(),
            tensors[0]._dtype.clone(),
            tensors[0]._device.clone(),
            tensors[0]._grad.clone(),
        ))
    }

    /// Statically stacks `self` with `other` along `Axis`.
    pub fn stack<Axis>(
        &self,
        other: &Tensor<S, B, G>,
    ) -> Result<Tensor<<S as crate::shapes::stack::StackShape<Axis>>::Output, B, G>>
    where
        Axis: typenum::Unsigned,
        S: crate::shapes::stack::StackShape<Axis>,
        <<S as crate::shapes::stack::StackShape<Axis>>::Output as Shape>::Field:
            core::default::Default,
    {
        let dim = Axis::USIZE;
        let inner = B::stack(&[&self.inner, &other.inner], dim)?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            core::default::Default::default(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Dynamically stacks `self` with `other` along `dim`.
    pub fn try_stack(&self, other: &Tensor<S, B, G>, dim: usize) -> Result<Tensor<Dyn, B, G>> {
        let inner = B::stack(&[&self.inner, &other.inner], dim)?;
        let mut out_shape = B::shape(&self.inner);
        out_shape.insert(dim, 2);
        Ok(Tensor::from_parts_unchecked(
            inner,
            <Dyn as Shape>::from_dyn(&out_shape).unwrap(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::{DType, Device};
    use crate::prelude::{KindleDType, KindleDevice};
    use alloc::vec;

    #[derive(Clone)]
    pub struct DummyOpsBackend<T: DType, D: Device>(core::marker::PhantomData<(T, D)>);
    impl<T: DType, D: Device> Backend for DummyOpsBackend<T, D> {
        fn shape(t: &Self::RawTensor) -> alloc::vec::Vec<usize> {
            t.clone()
        }

        fn conv1d(
            _t: &Self::RawTensor,
            _w: &Self::RawTensor,
            _b: Option<&Self::RawTensor>,
            _stride: usize,
            _padding: usize,
            _dilation: usize,
        ) -> Result<Self::RawTensor> {
            unimplemented!()
        }

        fn conv_transpose2d(
            _t: &Self::RawTensor,
            _w: &Self::RawTensor,
            _b: Option<&Self::RawTensor>,
            _stride: usize,
            _padding: usize,
            _output_padding: usize,
            _dilation: usize,
        ) -> Result<Self::RawTensor> {
            unimplemented!()
        }

        fn max_pool2d(
            _t: &Self::RawTensor,
            _kernel_size: (usize, usize),
            _stride: (usize, usize),
            _padding: (usize, usize),
            _dilation: (usize, usize),
        ) -> Result<Self::RawTensor> {
            unimplemented!()
        }

        fn avg_pool2d(
            _t: &Self::RawTensor,
            _kernel_size: (usize, usize),
            _stride: (usize, usize),
            _padding: (usize, usize),
        ) -> Result<Self::RawTensor> {
            unimplemented!()
        }

        fn embedding(_t: &Self::RawTensor, _w: &Self::RawTensor) -> Result<Self::RawTensor> {
            unimplemented!()
        }

        type Device = D;
        type DType = T;
        type BackendWithDType<NewT: DType> = DummyOpsBackend<NewT, D>; // Mock, won't actually change types
        type BackendWithDevice<NewD: Device> = DummyOpsBackend<T, NewD>;

        type RawTensor = alloc::vec::Vec<usize>;
        type RawVar = alloc::vec::Vec<usize>;
        type Grads = ();

        fn var_as_tensor(var: &Self::RawVar) -> Result<Self::RawTensor> {
            Ok(var.clone())
        }
        fn var_from_tensor(t: &Self::RawTensor) -> Result<Self::RawVar> {
            Ok(t.clone())
        }
        fn var_to_device(var: &Self::RawVar, _dev: &KindleDevice) -> Result<Self::RawVar> {
            Ok(var.clone())
        }
        fn var_zeros(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawVar> {
            Ok(shape.to_vec())
        }
        fn var_ones(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawVar> {
            Ok(shape.to_vec())
        }
        fn var_rand(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawVar> {
            Ok(shape.to_vec())
        }
        fn var_randn(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawVar> {
            Ok(shape.to_vec())
        }

        fn to_bytes(_t: &Self::RawTensor) -> Result<alloc::vec::Vec<u8>> {
            Ok(alloc::vec::Vec::new())
        }

        fn from_bytes(
            _bytes: &[u8],
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Ok(shape.to_vec())
        }

        fn zeros(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Ok(shape.to_vec())
        }
        fn ones(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Ok(shape.to_vec())
        }
        fn rand(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Ok(shape.to_vec())
        }
        fn randn(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Ok(shape.to_vec())
        }

        fn abs(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn relu(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn gelu(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn swish(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn softmax(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn neg(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn sqrt(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn exp(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn log(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn tanh(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn sigmoid(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }

        fn mul_scalar(t: &Self::RawTensor, _s: f64) -> Result<Self::RawTensor> {
            Ok(t.clone())
        }
        fn add_scalar(t: &Self::RawTensor, _s: f64) -> Result<Self::RawTensor> {
            Ok(t.clone())
        }

        fn sum_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn mean_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn max_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn min_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }

        fn sum_dim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn mean_dim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn max_dim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn min_dim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }

        fn sum_keepdim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn mean_keepdim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn max_keepdim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn min_keepdim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }

        fn stack(_t: &[&Self::RawTensor], _d: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn concat(_t: &[&Self::RawTensor], _d: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn layer_norm(
            _t: &Self::RawTensor,
            _w: &Self::RawTensor,
            _b: &Self::RawTensor,
            _e: f32,
        ) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn batch_norm(
            _t: &Self::RawTensor,
            _w: &Self::RawTensor,
            _b: &Self::RawTensor,
            _rm: &Self::RawTensor,
            _rv: &Self::RawTensor,
            _e: f32,
        ) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }

        fn tensor_to_device(_t: &Self::RawTensor, _dev: &KindleDevice) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn slice(_t: &Self::RawTensor, _ranges: &[(usize, usize)]) -> Result<Self::RawTensor> {
            unimplemented!()
        }

        fn to_dtype(_t: &Self::RawTensor, _dt: KindleDType) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }

        fn broadcast_as(_t: &Self::RawTensor, _s: &[usize]) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn broadcast_left(_t: &Self::RawTensor, _s: &[usize]) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }

        fn add(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn sub(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn mul(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn div(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn matmul(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn reshape(_t: &Self::RawTensor, _shape: &[usize]) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn transpose(_t: &Self::RawTensor, _d1: usize, _d2: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn flatten(_t: &Self::RawTensor, _s: usize, _e: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn narrow(
            _t: &Self::RawTensor,
            _dim: usize,
            _s: usize,
            _l: usize,
        ) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn squeeze(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }

        fn backward(_loss: &Self::RawTensor) -> Result<Self::Grads> {
            Ok(())
        }
        fn assign_var(var: &mut Self::RawVar, tensor: &Self::RawTensor) -> Result<()> {
            if var != tensor {
                return Err(crate::err::Error::ShapeMismatch {
                    op: "assign_var",
                    expected: var.clone(),
                    got: tensor.clone(),
                    msg: alloc::string::String::from("shape mismatch during assign_var"),
                });
            }
            *var = tensor.clone();
            Ok(())
        }
        fn get_grad(_var: &Self::RawVar, _grads: &Self::Grads) -> Result<Option<Self::RawTensor>> {
            unimplemented!()
        }

        fn conv2d(
            _t: &Self::RawTensor,
            _w: &Self::RawTensor,
            _b: Option<&Self::RawTensor>,
            _s: usize,
            _p: usize,
            _d: usize,
        ) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }

        fn format_tensor(t: &Self::RawTensor) -> alloc::string::String {
            alloc::format!("{:?}", t)
        }

        fn adaptive_avg_pool2d(
            _t: &Self::RawTensor,
            _output_size: (usize, usize),
        ) -> Result<Self::RawTensor> {
            unimplemented!()
        }

        fn mse_loss(
            _pred: &Self::RawTensor,
            _target: &Self::RawTensor,
            _reduction: crate::nn::Reduction,
        ) -> Result<Self::RawTensor> {
            unimplemented!()
        }
        fn l1_loss(
            _pred: &Self::RawTensor,
            _target: &Self::RawTensor,
            _reduction: crate::nn::Reduction,
        ) -> Result<Self::RawTensor> {
            unimplemented!()
        }
        fn bce_with_logits_loss(
            _pred: &Self::RawTensor,
            _target: &Self::RawTensor,
            _reduction: crate::nn::Reduction,
        ) -> Result<Self::RawTensor> {
            unimplemented!()
        }
        fn cross_entropy_loss(
            _pred: &Self::RawTensor,
            _target: &Self::RawTensor,
            _reduction: crate::nn::Reduction,
        ) -> Result<Self::RawTensor> {
            unimplemented!()
        }
    }

    #[test]
    fn test_tensor_ops() {
        let t1: Tensor<Dyn, DummyOpsBackend<f32, crate::prelude::Cpu>> =
            Tensor::zeros(vec![2, 2]).unwrap();
        let t2: Tensor<Dyn, DummyOpsBackend<f32, crate::prelude::Cpu>> =
            Tensor::ones(vec![2, 2]).unwrap();

        // Binary ops
        let _res_add = t1.add(&t2).unwrap();
        let _res_sub = t1.sub(&t2).unwrap();
        let _res_mul = t1.mul(&t2).unwrap();
        let _res_div = t1.div(&t2).unwrap();

        // Unary ops
        let _res_abs = t1.abs().unwrap();
        let _res_relu = t1.relu().unwrap();
        let _res_exp = t1.exp().unwrap();

        // Scalar ops
        let _res_muls = t1.mul_scalar(2.0).unwrap();

        // Slicing
        let _res_slice = t1
            .dyn_slice(&[
                crate::tensor::ops::IndexSpec::All,
                crate::tensor::ops::IndexSpec::Index(0),
            ])
            .unwrap();
    }
}

pub fn try_stack_tensors<S: Shape + DynShape, B: Backend, G: crate::tensor::grad::RequiresGrad>(
    tensors: &[&Tensor<S, B, G>],
    dim: usize,
) -> Result<Tensor<Dyn, B, G>>
where
    G::Field: Clone,
{
    if tensors.is_empty() {
        return Err(crate::prelude::Error::ShapeMismatch {
            op: "stack_tensors",
            expected: alloc::vec![],
            got: alloc::vec![],
            msg: alloc::string::String::from("Cannot stack empty list of tensors"),
        });
    }
    let raw_tensors: alloc::vec::Vec<&B::RawTensor> = tensors.iter().map(|t| &t.inner).collect();
    let inner = B::stack(&raw_tensors, dim)?;
    let mut shape = S::dims(&tensors[0]._shape).as_ref().to_vec();
    shape.insert(dim, tensors.len());
    Ok(Tensor {
        inner,
        _shape: shape,
        _dtype: tensors[0]._dtype.clone(),
        _device: tensors[0]._device.clone(),
        _grad: tensors[0]._grad.clone(),
    })
}
