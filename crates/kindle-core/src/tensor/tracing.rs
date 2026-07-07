use crate::prelude::*;
use crate::graph::{Graph, OpType, ValueId};
use std::cell::RefCell;

thread_local! {
    pub static TRACING_GRAPH: RefCell<Graph> = RefCell::new(Graph::new());
}

pub fn extract_graph() -> Graph {
    TRACING_GRAPH.with(|g| {
        let mut b = g.borrow_mut();
        std::mem::take(&mut *b)
    })
}

#[derive(Clone)]
pub struct TracingBackend<B: Backend> {
    _marker: core::marker::PhantomData<B>,
}

#[derive(Clone)]
pub struct TracingTensor<T> {
    pub inner: T,
    pub value_id: ValueId,
}

#[derive(Clone)]
pub struct TracingVar<V> {
    pub inner: V,
    pub value_id: ValueId,
}

impl<B: Backend> TracingBackend<B> {
    // A helper for binary ops
    fn trace_binary(op: OpType, lhs: &TracingTensor<B::RawTensor>, rhs: &TracingTensor<B::RawTensor>, inner_res: &B::RawTensor) -> TracingTensor<B::RawTensor> {
        let shape = B::shape(inner_res);
        let value_id = TRACING_GRAPH.with(|g| {
            let mut g = g.borrow_mut();
            let out_id = g.add_value(shape, KindleDType::F32, None); // default F32 for now
            g.add_node(op, vec![lhs.value_id, rhs.value_id], vec![out_id], std::collections::HashMap::new());
            out_id
        });
        TracingTensor {
            inner: inner_res.clone(),
            value_id,
        }
    }

    // A helper for unary ops
    fn trace_unary(op: OpType, t: &TracingTensor<B::RawTensor>, inner_res: &B::RawTensor) -> TracingTensor<B::RawTensor> {
        let shape = B::shape(inner_res);
        let value_id = TRACING_GRAPH.with(|g| {
            let mut g = g.borrow_mut();
            let out_id = g.add_value(shape, KindleDType::F32, None);
            g.add_node(op, vec![t.value_id], vec![out_id], std::collections::HashMap::new());
            out_id
        });
        TracingTensor {
            inner: inner_res.clone(),
            value_id,
        }
    }
}

impl<B: Backend> Backend for TracingBackend<B> {
    type Device = B::Device;
    type DType = B::DType;
    type BackendWithDType<NewT: crate::tensor::dtype::DType> = TracingBackend<B::BackendWithDType<NewT>>;
    type BackendWithDevice<NewD: crate::tensor::device::Device> = TracingBackend<B::BackendWithDevice<NewD>>;

    type RawTensor = TracingTensor<B::RawTensor>;
    type RawVar = TracingVar<B::RawVar>;
    type Grads = B::Grads; // For now

    fn shape(t: &Self::RawTensor) -> alloc::vec::Vec<usize> {
        B::shape(&t.inner)
    }

    fn format_tensor(t: &Self::RawTensor) -> alloc::string::String {
        B::format_tensor(&t.inner)
    }

    fn var_as_tensor(var: &Self::RawVar) -> Result<Self::RawTensor> {
        let inner = B::var_as_tensor(&var.inner)?;
        Ok(TracingTensor {
            inner,
            value_id: var.value_id,
        })
    }

    fn var_from_tensor(t: &Self::RawTensor) -> Result<Self::RawVar> {
        let inner = B::var_from_tensor(&t.inner)?;
        Ok(TracingVar {
            inner,
            value_id: t.value_id,
        })
    }

    fn zeros(shape: &[usize], dtype: KindleDType, device: &KindleDevice) -> Result<Self::RawTensor> {
        let inner = B::zeros(shape, dtype, device)?;
        let value_id = TRACING_GRAPH.with(|g| g.borrow_mut().add_value(shape.to_vec(), dtype, None));
        Ok(TracingTensor { inner, value_id })
    }

    fn ones(shape: &[usize], dtype: KindleDType, device: &KindleDevice) -> Result<Self::RawTensor> {
        let inner = B::ones(shape, dtype, device)?;
        let value_id = TRACING_GRAPH.with(|g| g.borrow_mut().add_value(shape.to_vec(), dtype, None));
        Ok(TracingTensor { inner, value_id })
    }

    fn rand(shape: &[usize], dtype: KindleDType, device: &KindleDevice) -> Result<Self::RawTensor> {
        let inner = B::rand(shape, dtype, device)?;
        let value_id = TRACING_GRAPH.with(|g| g.borrow_mut().add_value(shape.to_vec(), dtype, None));
        Ok(TracingTensor { inner, value_id })
    }

    fn randn(shape: &[usize], dtype: KindleDType, device: &KindleDevice) -> Result<Self::RawTensor> {
        let inner = B::randn(shape, dtype, device)?;
        let value_id = TRACING_GRAPH.with(|g| g.borrow_mut().add_value(shape.to_vec(), dtype, None));
        Ok(TracingTensor { inner, value_id })
    }

    fn var_zeros(shape: &[usize], dtype: KindleDType, device: &KindleDevice) -> Result<Self::RawVar> {
        let inner = B::var_zeros(shape, dtype, device)?;
        let value_id = TRACING_GRAPH.with(|g| g.borrow_mut().add_value(shape.to_vec(), dtype, None));
        Ok(TracingVar { inner, value_id })
    }

    fn var_ones(shape: &[usize], dtype: KindleDType, device: &KindleDevice) -> Result<Self::RawVar> {
        let inner = B::var_ones(shape, dtype, device)?;
        let value_id = TRACING_GRAPH.with(|g| g.borrow_mut().add_value(shape.to_vec(), dtype, None));
        Ok(TracingVar { inner, value_id })
    }

    fn var_rand(shape: &[usize], dtype: KindleDType, device: &KindleDevice) -> Result<Self::RawVar> {
        let inner = B::var_rand(shape, dtype, device)?;
        let value_id = TRACING_GRAPH.with(|g| g.borrow_mut().add_value(shape.to_vec(), dtype, None));
        Ok(TracingVar { inner, value_id })
    }

    fn tensor_to_device(t: &Self::RawTensor, device: &KindleDevice) -> Result<Self::RawTensor> {
        let inner = B::tensor_to_device(&t.inner, device)?;
        Ok(TracingTensor { inner, value_id: t.value_id })
    }

    fn var_to_device(var: &Self::RawVar, device: &KindleDevice) -> Result<Self::RawVar> {
        let inner = B::var_to_device(&var.inner, device)?;
        Ok(TracingVar { inner, value_id: var.value_id })
    }

    fn assign_var(var: &mut Self::RawVar, tensor: &Self::RawTensor) -> Result<()> {
        B::assign_var(&mut var.inner, &tensor.inner)
    }

    fn var_randn(shape: &[usize], dtype: KindleDType, device: &KindleDevice) -> Result<Self::RawVar> {
        let inner = B::var_randn(shape, dtype, device)?;
        let value_id = TRACING_GRAPH.with(|g| g.borrow_mut().add_value(shape.to_vec(), dtype, None));
        Ok(TracingVar { inner, value_id })
    }

    fn relu(t: &Self::RawTensor) -> Result<Self::RawTensor> {
        let inner = B::relu(&t.inner)?;
        Ok(Self::trace_unary(OpType::Relu, t, &inner))
    }

    fn gelu(t: &Self::RawTensor) -> Result<Self::RawTensor> {
        let inner = B::gelu(&t.inner)?;
        Ok(Self::trace_unary(OpType::Gelu, t, &inner))
    }

    fn abs(t: &Self::RawTensor) -> Result<Self::RawTensor> {
        let inner = B::abs(&t.inner)?;
        Ok(Self::trace_unary(OpType::Relu, t, &inner)) // Replace with Abs later
    }

    fn exp(t: &Self::RawTensor) -> Result<Self::RawTensor> {
        let inner = B::exp(&t.inner)?;
        Ok(Self::trace_unary(OpType::Relu, t, &inner))
    }

    fn neg(t: &Self::RawTensor) -> Result<Self::RawTensor> {
        let inner = B::neg(&t.inner)?;
        Ok(Self::trace_unary(OpType::Relu, t, &inner))
    }

    fn sqrt(t: &Self::RawTensor) -> Result<Self::RawTensor> {
        let inner = B::sqrt(&t.inner)?;
        Ok(Self::trace_unary(OpType::Relu, t, &inner))
    }

    fn log(t: &Self::RawTensor) -> Result<Self::RawTensor> {
        let inner = B::log(&t.inner)?;
        Ok(Self::trace_unary(OpType::Relu, t, &inner))
    }

    fn tanh(t: &Self::RawTensor) -> Result<Self::RawTensor> {
        let inner = B::tanh(&t.inner)?;
        Ok(Self::trace_unary(OpType::Relu, t, &inner))
    }

    fn sigmoid(t: &Self::RawTensor) -> Result<Self::RawTensor> {
        let inner = B::sigmoid(&t.inner)?;
        Ok(Self::trace_unary(OpType::Relu, t, &inner))
    }

    fn swish(t: &Self::RawTensor) -> Result<Self::RawTensor> {
        let inner = B::swish(&t.inner)?;
        Ok(Self::trace_unary(OpType::Relu, t, &inner))
    }

    fn softmax(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> {
        let inner = B::softmax(&t.inner, dim)?;
        let shape = B::shape(&inner);
        let value_id = TRACING_GRAPH.with(|g| {
            let mut g = g.borrow_mut();
            let out_id = g.add_value(shape, KindleDType::F32, None);
            let mut attrs = std::collections::HashMap::new();
            attrs.insert("axis".to_string(), crate::graph::AttributeValue::Int(dim as i64));
            g.add_node(OpType::Softmax, vec![t.value_id], vec![out_id], attrs);
            out_id
        });
        Ok(TracingTensor { inner, value_id })
    }

    fn mul_scalar(t: &Self::RawTensor, scalar: f64) -> Result<Self::RawTensor> {
        let inner = B::mul_scalar(&t.inner, scalar)?;
        Ok(Self::trace_unary(OpType::MulScalar, t, &inner))
    }

    fn add_scalar(t: &Self::RawTensor, scalar: f64) -> Result<Self::RawTensor> {
        let inner = B::add_scalar(&t.inner, scalar)?;
        Ok(Self::trace_unary(OpType::AddScalar, t, &inner))
    }

    fn sum_all(t: &Self::RawTensor) -> Result<Self::RawTensor> {
        let inner = B::sum_all(&t.inner)?;
        Ok(Self::trace_unary(OpType::SumAll, t, &inner))
    }

    fn mean_all(t: &Self::RawTensor) -> Result<Self::RawTensor> {
        let inner = B::mean_all(&t.inner)?;
        Ok(Self::trace_unary(OpType::MeanAll, t, &inner))
    }

    fn max_all(t: &Self::RawTensor) -> Result<Self::RawTensor> {
        let inner = B::max_all(&t.inner)?;
        Ok(Self::trace_unary(OpType::MaxAll, t, &inner))
    }

    fn min_all(t: &Self::RawTensor) -> Result<Self::RawTensor> {
        let inner = B::min_all(&t.inner)?;
        Ok(Self::trace_unary(OpType::MinAll, t, &inner))
    }

    fn sum_dim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> {
        let inner = B::sum_dim(&t.inner, dim)?;
        Ok(Self::trace_unary(OpType::SumDim, t, &inner))
    }

    fn sum_keepdim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> {
        let inner = B::sum_keepdim(&t.inner, dim)?;
        Ok(Self::trace_unary(OpType::SumDim, t, &inner))
    }

    fn mean_dim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> {
        let inner = B::mean_dim(&t.inner, dim)?;
        Ok(Self::trace_unary(OpType::MeanDim, t, &inner))
    }

    fn mean_keepdim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> {
        let inner = B::mean_keepdim(&t.inner, dim)?;
        Ok(Self::trace_unary(OpType::MeanDim, t, &inner))
    }

    fn max_dim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> {
        let inner = B::max_dim(&t.inner, dim)?;
        Ok(Self::trace_unary(OpType::MaxDim, t, &inner))
    }

    fn max_keepdim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> {
        let inner = B::max_keepdim(&t.inner, dim)?;
        Ok(Self::trace_unary(OpType::MaxDim, t, &inner))
    }

    fn min_dim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> {
        let inner = B::min_dim(&t.inner, dim)?;
        Ok(Self::trace_unary(OpType::MinDim, t, &inner))
    }

    fn min_keepdim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> {
        let inner = B::min_keepdim(&t.inner, dim)?;
        Ok(Self::trace_unary(OpType::MinDim, t, &inner))
    }

    fn add(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
        let inner = B::add(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(OpType::Add, lhs, rhs, &inner))
    }

    fn sub(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
        let inner = B::sub(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(OpType::Sub, lhs, rhs, &inner))
    }

    fn mul(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
        let inner = B::mul(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(OpType::Mul, lhs, rhs, &inner))
    }

    fn div(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
        let inner = B::div(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(OpType::Div, lhs, rhs, &inner))
    }

    fn matmul(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
        let inner = B::matmul(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(OpType::MatMul, lhs, rhs, &inner))
    }

    fn broadcast_as(t: &Self::RawTensor, shape: &[usize]) -> Result<Self::RawTensor> {
        let inner = B::broadcast_as(&t.inner, shape)?;
        Ok(Self::trace_unary(OpType::Broadcast, t, &inner))
    }

    fn reshape(t: &Self::RawTensor, shape: &[usize]) -> Result<Self::RawTensor> {
        let inner = B::reshape(&t.inner, shape)?;
        let shape_out = B::shape(&inner);
        let value_id = TRACING_GRAPH.with(|g| {
            let mut g = g.borrow_mut();
            let out_id = g.add_value(shape_out.clone(), KindleDType::F32, None);
            
            // Add reshape parameters as a constant value
            let shape_val_id = g.add_value(vec![shape.len()], KindleDType::I64, None);
            let mut bytes = Vec::new();
            for &s in shape {
                bytes.extend_from_slice(&(s as i64).to_le_bytes());
            }
            g.initializers.insert(shape_val_id, bytes);
            
            g.add_node(OpType::Reshape, vec![t.value_id, shape_val_id], vec![out_id], std::collections::HashMap::new());
            out_id
        });
        Ok(TracingTensor { inner, value_id })
    }

    fn transpose(t: &Self::RawTensor, dim1: usize, dim2: usize) -> Result<Self::RawTensor> {
        let inner = B::transpose(&t.inner, dim1, dim2)?;
        let shape_out = B::shape(&inner);
        let value_id = TRACING_GRAPH.with(|g| {
            let mut g = g.borrow_mut();
            let out_id = g.add_value(shape_out.clone(), KindleDType::F32, None);
            let mut attrs = std::collections::HashMap::new();
            // simple perm vector building for ONNX
            let mut perm: Vec<i64> = (0..shape_out.len() as i64).collect();
            perm.swap(dim1, dim2);
            attrs.insert("perm".to_string(), crate::graph::AttributeValue::Ints(perm));
            g.add_node(OpType::Transpose, vec![t.value_id], vec![out_id], attrs);
            out_id
        });
        Ok(TracingTensor { inner, value_id })
    }

    fn narrow(t: &Self::RawTensor, dim: usize, start: usize, len: usize) -> Result<Self::RawTensor> {
        let inner = B::narrow(&t.inner, dim, start, len)?;
        Ok(Self::trace_unary(OpType::Narrow, t, &inner))
    }

    fn concat(tensors: &[&Self::RawTensor], dim: usize) -> Result<Self::RawTensor> {
        let inners: Vec<&B::RawTensor> = tensors.iter().map(|t| &t.inner).collect();
        let inner = B::concat(&inners, dim)?;
        let shape_out = B::shape(&inner);
        let value_id = TRACING_GRAPH.with(|g| {
            let mut g = g.borrow_mut();
            let out_id = g.add_value(shape_out, KindleDType::F32, None);
            let inputs = tensors.iter().map(|t| t.value_id).collect();
            let mut attrs = std::collections::HashMap::new();
            attrs.insert("axis".to_string(), crate::graph::AttributeValue::Int(dim as i64));
            g.add_node(OpType::Concat, inputs, vec![out_id], attrs);
            out_id
        });
        Ok(TracingTensor { inner, value_id })
    }

    fn stack(tensors: &[&Self::RawTensor], dim: usize) -> Result<Self::RawTensor> {
        let inners: Vec<&B::RawTensor> = tensors.iter().map(|t| &t.inner).collect();
        let inner = B::stack(&inners, dim)?;
        let shape_out = B::shape(&inner);
        let value_id = TRACING_GRAPH.with(|g| {
            let mut g = g.borrow_mut();
            let out_id = g.add_value(shape_out, KindleDType::F32, None);
            let inputs = tensors.iter().map(|t| t.value_id).collect();
            let mut attrs = std::collections::HashMap::new();
            attrs.insert("axis".to_string(), crate::graph::AttributeValue::Int(dim as i64));
            g.add_node(OpType::Stack, inputs, vec![out_id], attrs);
            out_id
        });
        Ok(TracingTensor { inner, value_id })
    }

    fn conv1d(
        x: &Self::RawTensor,
        weight: &Self::RawTensor,
        bias: Option<&Self::RawTensor>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Result<Self::RawTensor> {
        let inner_bias = bias.map(|b| &b.inner);
        let inner = B::conv1d(&x.inner, &weight.inner, inner_bias, stride, padding, dilation)?;
        let shape_out = B::shape(&inner);
        let value_id = TRACING_GRAPH.with(|g| {
            let mut g = g.borrow_mut();
            let out_id = g.add_value(shape_out, KindleDType::F32, None);
            let mut inputs = vec![x.value_id, weight.value_id];
            if let Some(b) = bias {
                inputs.push(b.value_id);
            }
            let mut attrs = std::collections::HashMap::new();
            attrs.insert("strides".to_string(), crate::graph::AttributeValue::Ints(vec![stride as i64]));
            attrs.insert("pads".to_string(), crate::graph::AttributeValue::Ints(vec![padding as i64, padding as i64]));
            attrs.insert("dilations".to_string(), crate::graph::AttributeValue::Ints(vec![dilation as i64]));
            
            g.add_node(OpType::Conv1d, inputs, vec![out_id], attrs);
            out_id
        });
        Ok(TracingTensor { inner, value_id })
    }

    fn conv2d(
        x: &Self::RawTensor,
        weight: &Self::RawTensor,
        bias: Option<&Self::RawTensor>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Result<Self::RawTensor> {
        let inner_bias = bias.map(|b| &b.inner);
        let inner = B::conv2d(&x.inner, &weight.inner, inner_bias, stride, padding, dilation)?;
        let shape_out = B::shape(&inner);
        let value_id = TRACING_GRAPH.with(|g| {
            let mut g = g.borrow_mut();
            let out_id = g.add_value(shape_out, KindleDType::F32, None);
            let mut inputs = vec![x.value_id, weight.value_id];
            if let Some(b) = bias {
                inputs.push(b.value_id);
            }
            let mut attrs = std::collections::HashMap::new();
            attrs.insert("strides".to_string(), crate::graph::AttributeValue::Ints(vec![stride as i64, stride as i64]));
            attrs.insert("pads".to_string(), crate::graph::AttributeValue::Ints(vec![padding as i64, padding as i64, padding as i64, padding as i64]));
            attrs.insert("dilations".to_string(), crate::graph::AttributeValue::Ints(vec![dilation as i64, dilation as i64]));
            
            g.add_node(OpType::Conv2d, inputs, vec![out_id], attrs);
            out_id
        });
        Ok(TracingTensor { inner, value_id })
    }

    fn max_pool2d(
        x: &Self::RawTensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<Self::RawTensor> {
        let inner = B::max_pool2d(&x.inner, kernel_size, stride, padding, dilation)?;
        Ok(Self::trace_unary(OpType::MaxPool2d, x, &inner))
    }

    fn avg_pool2d(
        x: &Self::RawTensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self::RawTensor> {
        let inner = B::avg_pool2d(&x.inner, kernel_size, stride, padding)?;
        Ok(Self::trace_unary(OpType::AvgPool2d, x, &inner))
    }

    fn adaptive_avg_pool2d(
        x: &Self::RawTensor,
        output_size: (usize, usize),
    ) -> Result<Self::RawTensor> {
        let inner = B::adaptive_avg_pool2d(&x.inner, output_size)?;
        Ok(Self::trace_unary(OpType::AdaptiveAvgPool2d, x, &inner))
    }

    fn backward(loss: &Self::RawTensor) -> Result<Self::Grads> {
        B::backward(&loss.inner)
    }

    fn get_grad(_var: &Self::RawVar, _grads: &Self::Grads) -> Result<Option<Self::RawTensor>> {
        Ok(None)
    }

    fn from_bytes(
        bytes: &[u8],
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<Self::RawTensor> {
        let inner = B::from_bytes(bytes, shape, dtype, device)?;
        let value_id = TRACING_GRAPH.with(|g| {
            let mut g = g.borrow_mut();
            let id = g.add_value(shape.to_vec(), dtype, None);
            g.initializers.insert(id, bytes.to_vec());
            id
        });
        Ok(TracingTensor { inner, value_id })
    }

    fn to_bytes(t: &Self::RawTensor) -> Result<alloc::vec::Vec<u8>> {
        B::to_bytes(&t.inner)
    }

    fn slice(t: &Self::RawTensor, ranges: &[(usize, usize)]) -> Result<Self::RawTensor> {
        let inner = B::slice(&t.inner, ranges)?;
        Ok(Self::trace_unary(OpType::Slice, t, &inner))
    }

    fn to_dtype(t: &Self::RawTensor, dtype: KindleDType) -> Result<Self::RawTensor> {
        let inner = B::to_dtype(&t.inner, dtype)?;
        Ok(Self::trace_unary(OpType::ToDtype, t, &inner))
    }

    fn cross_entropy_loss(
        logits: &Self::RawTensor,
        targets: &Self::RawTensor,
        reduction: crate::nn::loss::Reduction,
    ) -> Result<Self::RawTensor> {
        let inner = B::cross_entropy_loss(&logits.inner, &targets.inner, reduction)?;
        Ok(Self::trace_binary(OpType::CrossEntropyLoss, logits, targets, &inner))
    }

    fn mse_loss(
        predictions: &Self::RawTensor,
        targets: &Self::RawTensor,
        reduction: crate::nn::loss::Reduction,
    ) -> Result<Self::RawTensor> {
        let inner = B::mse_loss(&predictions.inner, &targets.inner, reduction)?;
        Ok(Self::trace_binary(OpType::MseLoss, predictions, targets, &inner))
    }

    fn l1_loss(
        predictions: &Self::RawTensor,
        targets: &Self::RawTensor,
        reduction: crate::nn::loss::Reduction,
    ) -> Result<Self::RawTensor> {
        let inner = B::l1_loss(&predictions.inner, &targets.inner, reduction)?;
        Ok(Self::trace_binary(OpType::L1Loss, predictions, targets, &inner))
    }

    fn bce_with_logits_loss(
        logits: &Self::RawTensor,
        targets: &Self::RawTensor,
        reduction: crate::nn::loss::Reduction,
    ) -> Result<Self::RawTensor> {
        let inner = B::bce_with_logits_loss(&logits.inner, &targets.inner, reduction)?;
        Ok(Self::trace_binary(OpType::BceWithLogitsLoss, logits, targets, &inner))
    }

    fn embedding(
        weight: &Self::RawTensor,
        indices: &Self::RawTensor,
    ) -> Result<Self::RawTensor> {
        let inner = B::embedding(&weight.inner, &indices.inner)?;
        Ok(Self::trace_binary(OpType::Embedding, weight, indices, &inner))
    }

    fn layer_norm(
        x: &Self::RawTensor,
        weight: &Self::RawTensor,
        bias: &Self::RawTensor,
        eps: f32,
    ) -> Result<Self::RawTensor> {
        let inner = B::layer_norm(&x.inner, &weight.inner, &bias.inner, eps)?;
        Ok(Self::trace_unary(OpType::LayerNorm, x, &inner))
    }

    fn batch_norm(
        t: &Self::RawTensor,
        w: &Self::RawTensor,
        b: &Self::RawTensor,
        rm: &Self::RawTensor,
        rv: &Self::RawTensor,
        e: f32,
    ) -> Result<Self::RawTensor> {
        let inner = B::batch_norm(&t.inner, &w.inner, &b.inner, &rm.inner, &rv.inner, e)?;
        Ok(Self::trace_unary(OpType::BatchNorm, t, &inner))
    }

    fn flatten(t: &Self::RawTensor, start_dim: usize, end_dim: usize) -> Result<Self::RawTensor> {
        let inner = B::flatten(&t.inner, start_dim, end_dim)?;
        Ok(Self::trace_unary(OpType::Reshape, t, &inner))
    }

    fn squeeze(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> {
        let inner = B::squeeze(&t.inner, dim)?;
        Ok(Self::trace_unary(OpType::Reshape, t, &inner))
    }
    
    fn broadcast_left(t: &Self::RawTensor, shape: &[usize]) -> Result<Self::RawTensor> {
        let inner = B::broadcast_left(&t.inner, shape)?;
        Ok(Self::trace_unary(OpType::Broadcast, t, &inner))
    }
    
    fn conv_transpose2d(
        t: &Self::RawTensor,
        w: &Self::RawTensor,
        b: Option<&Self::RawTensor>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
    ) -> Result<Self::RawTensor> {
        let inner_b = b.map(|b| &b.inner);
        let inner = B::conv_transpose2d(&t.inner, &w.inner, inner_b, stride, padding, output_padding, dilation)?;
        Ok(Self::trace_unary(OpType::ConvTranspose2d, t, &inner))
    }
}
