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

    #[test]
    fn test_tensor_ops() {
        let t1: Tensor<Dyn, crate::tensor::backend::dummy::DummyBackend<f32, crate::prelude::Cpu>> =
            Tensor::zeros(vec![2, 2]).unwrap();
        let t2: Tensor<Dyn, crate::tensor::backend::dummy::DummyBackend<f32, crate::prelude::Cpu>> =
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
                crate::tensor::ops::IndexSpec::All, (crate::tensor::ops::IndexSpec::Index(0).into()),
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
