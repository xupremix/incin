use crate::graph::{Graph, OpType, ValueId};
use crate::prelude::*;
use crate::tensor::backend::*;
// removed RefCell
use spin::{Mutex, Lazy};

pub static TRACING_GRAPH: Lazy<Mutex<Graph>> = Lazy::new(|| Mutex::new(Graph::new()));

pub fn extract_graph() -> Graph {
    let mut b = TRACING_GRAPH.lock();
    core::mem::take(&mut *b)
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
    fn trace_binary<K1: super::dtype::DType, K2: super::dtype::DType, KOut: super::dtype::DType>(
        op: OpType,
        lhs: &TracingTensor<B::Storage<K1>>,
        rhs: &TracingTensor<B::Storage<K2>>,
        inner_res: &B::Storage<KOut>,
    ) -> TracingTensor<B::Storage<KOut>> {
        let shape = B::shape(inner_res);
        let value_id = {
            let mut g = TRACING_GRAPH.lock();
            let out_id = g.add_value(shape, KindleDType::F32, None); // default F32 for now
            g.add_node(
                op,
                vec![lhs.value_id, rhs.value_id],
                vec![out_id],
                hashbrown::HashMap::new(),
            );
            out_id
        };
        TracingTensor {
            inner: inner_res.clone(),
            value_id,
        }
    }

    // A helper for unary ops
    fn trace_unary<K: super::dtype::DType, KOut: super::dtype::DType>(
        op: OpType,
        t: &TracingTensor<B::Storage<K>>,
        inner_res: &B::Storage<KOut>,
    ) -> TracingTensor<B::Storage<KOut>> {
        let shape = B::shape(inner_res);
        let value_id = {
            let mut g = TRACING_GRAPH.lock();
            let out_id = g.add_value(shape, KindleDType::F32, None);
            g.add_node(
                op,
                vec![t.value_id],
                vec![out_id],
                hashbrown::HashMap::new(),
            );
            out_id
        };
        TracingTensor {
            inner: inner_res.clone(),
            value_id,
        }
    }
}

impl<B: Backend> Backend for TracingBackend<B> {
    type Storage<K: super::dtype::DType> = TracingTensor<B::Storage<K>>;
    type Device = B::Device;
    type FloatElem = B::FloatElem;
    type IntElem = B::IntElem;

    type BackendWithDevice<NewD: crate::tensor::device::Device> =
        TracingBackend<B::BackendWithDevice<NewD>>;

    type RawVar = TracingVar<B::RawVar>;
    type Grads = B::Grads;
    type InnerBackend = B::InnerBackend;

    fn shape<K: super::dtype::DType>(t: &<Self as Backend>::Storage<K>) -> alloc::vec::Vec<usize> {
        B::shape(&t.inner)
    }

    fn format_tensor_display<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> alloc::string::String {
        B::format_tensor_display(&t.inner)
    }
    fn format_tensor_debug<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> alloc::string::String {
        B::format_tensor_debug(&t.inner)
    }

    fn var_as_tensor<K: super::dtype::DType>(
        var: &<Self as Backend>::RawVar,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::var_as_tensor(&var.inner)?;
        Ok(TracingTensor {
            inner,
            value_id: var.value_id,
        })
    }

    fn var_from_tensor<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::RawVar> {
        let inner = B::var_from_tensor(&t.inner)?;
        Ok(TracingVar {
            inner,
            value_id: t.value_id,
        })
    }

    fn var_to_device(
        var: &<Self as Backend>::RawVar,
        device: &KindleDevice,
    ) -> Result<<Self as Backend>::RawVar> {
        let inner = B::var_to_device(&var.inner, device)?;
        Ok(TracingVar {
            inner,
            value_id: var.value_id,
        })
    }

    fn assign_var<K: super::dtype::DType>(
        var: &mut <Self as Backend>::RawVar,
        tensor: &<Self as Backend>::Storage<K>,
    ) -> Result<()> {
        B::assign_var(&mut var.inner, &tensor.inner)
    }

    fn backward<K: super::dtype::DType>(
        loss: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Grads> {
        B::backward(&loss.inner)
    }

    fn backward_with_nan_check<K: super::dtype::DType>(
        loss: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Grads> {
        B::backward_with_nan_check(&loss.inner)
    }

    fn get_grad<K: super::dtype::DType>(
        _var: &<Self as Backend>::Storage<K>,
        _grads: &<Self as Backend>::Grads,
    ) -> Result<Option<<Self as Backend>::Storage<K>>> {
        Ok(None)
    }

    fn from_bytes<K: super::dtype::DType>(
        bytes: &[u8],
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::from_bytes(bytes, shape, dtype, device)?;
        let value_id = {
            let mut g = TRACING_GRAPH.lock();
            let id = g.add_value(shape.to_vec(), KindleDType::F32, None);
            g.initializers.insert(id, bytes.to_vec());
            id
        };
        Ok(TracingTensor { inner, value_id })
    }

    fn to_bytes<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<alloc::vec::Vec<u8>> {
        B::to_bytes(&t.inner)
    }
}

impl<B: Backend> CreationOps<Self> for TracingBackend<B> {
    fn zeros<K: super::dtype::DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::zeros(shape, dtype, device)?;
        let value_id =
            TRACING_GRAPH.lock().add_value(shape.to_vec(), dtype, None);
        Ok(TracingTensor { inner, value_id })
    }

    fn ones<K: super::dtype::DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::ones(shape, dtype, device)?;
        let value_id =
            TRACING_GRAPH.lock().add_value(shape.to_vec(), dtype, None);
        Ok(TracingTensor { inner, value_id })
    }

    fn rand<K: super::dtype::DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::rand(shape, dtype, device)?;
        let value_id =
            TRACING_GRAPH.lock().add_value(shape.to_vec(), dtype, None);
        Ok(TracingTensor { inner, value_id })
    }

    fn randn<K: super::dtype::DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::randn(shape, dtype, device)?;
        let value_id =
            TRACING_GRAPH.lock().add_value(shape.to_vec(), dtype, None);
        Ok(TracingTensor { inner, value_id })
    }

    fn var_zeros<K: super::dtype::DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<<Self as Backend>::RawVar> {
        let inner = B::var_zeros::<K>(shape, dtype, device)?;
        let value_id =
            TRACING_GRAPH.lock().add_value(shape.to_vec(), dtype, None);
        Ok(TracingVar { inner, value_id })
    }

    fn var_ones<K: super::dtype::DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<<Self as Backend>::RawVar> {
        let inner = B::var_ones::<K>(shape, dtype, device)?;
        let value_id =
            TRACING_GRAPH.lock().add_value(shape.to_vec(), dtype, None);
        Ok(TracingVar { inner, value_id })
    }

    fn var_rand<K: super::dtype::DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<<Self as Backend>::RawVar> {
        let inner = B::var_rand::<K>(shape, dtype, device)?;
        let value_id =
            TRACING_GRAPH.lock().add_value(shape.to_vec(), dtype, None);
        Ok(TracingVar { inner, value_id })
    }

    fn var_randn<K: super::dtype::DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<<Self as Backend>::RawVar> {
        let inner = B::var_randn::<K>(shape, dtype, device)?;
        let value_id =
            TRACING_GRAPH.lock().add_value(shape.to_vec(), dtype, None);
        Ok(TracingVar { inner, value_id })
    }

    fn tensor_to_device<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
        device: &KindleDevice,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::tensor_to_device(&t.inner, device)?;
        Ok(TracingTensor {
            inner,
            value_id: t.value_id,
        })
    }
}

impl<B: Backend> NumericOps<Self> for TracingBackend<B> {
    fn add<K: super::dtype::DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::add(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(OpType::Add, lhs, rhs, &inner))
    }

    fn sub<K: super::dtype::DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::sub(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(OpType::Sub, lhs, rhs, &inner))
    }

    fn mul<K: super::dtype::DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::mul(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(OpType::Mul, lhs, rhs, &inner))
    }

    fn div<K: super::dtype::DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::div(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(OpType::Div, lhs, rhs, &inner))
    }
}

impl<B: Backend> FloatOps<Self> for TracingBackend<B> {
    fn add_scalar_float<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::add_scalar_float(&t.inner, scalar)?;
        Ok(Self::trace_unary(OpType::AddScalar, t, &inner))
    }

    fn mul_scalar_float<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::mul_scalar_float(&t.inner, scalar)?;
        Ok(Self::trace_unary(OpType::MulScalar, t, &inner))
    }

    fn relu<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::relu(&t.inner)?;
        Ok(Self::trace_unary(OpType::Relu, t, &inner))
    }

    fn gelu<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::gelu(&t.inner)?;
        Ok(Self::trace_unary(OpType::Gelu, t, &inner))
    }

    fn abs<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::abs(&t.inner)?;
        Ok(Self::trace_unary(OpType::Relu, t, &inner))
    }

    fn exp<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::exp(&t.inner)?;
        Ok(Self::trace_unary(OpType::Relu, t, &inner))
    }

    fn neg<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::neg(&t.inner)?;
        Ok(Self::trace_unary(OpType::Relu, t, &inner))
    }

    fn sqrt<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::sqrt(&t.inner)?;
        Ok(Self::trace_unary(OpType::Relu, t, &inner))
    }

    fn log<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::log(&t.inner)?;
        Ok(Self::trace_unary(OpType::Relu, t, &inner))
    }

    fn tanh<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::tanh(&t.inner)?;
        Ok(Self::trace_unary(OpType::Relu, t, &inner))
    }

    fn sigmoid<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::sigmoid(&t.inner)?;
        Ok(Self::trace_unary(OpType::Relu, t, &inner))
    }

    fn swish<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::swish(&t.inner)?;
        Ok(Self::trace_unary(OpType::Relu, t, &inner))
    }

    fn softmax<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::softmax(&t.inner, dim)?;
        let shape = B::shape(&inner);
        let value_id = {
            let mut g = TRACING_GRAPH.lock();
            let out_id = g.add_value(shape, KindleDType::F32, None);
            let mut attrs = hashbrown::HashMap::new();
            attrs.insert(
                alloc::string::String::from("axis"),
                crate::graph::AttributeValue::Int(dim as i64),
            );
            g.add_node(OpType::Softmax, vec![t.value_id], vec![out_id], attrs);
            out_id
        };
        Ok(TracingTensor { inner, value_id })
    }
}

impl<B: Backend> ReductionOps<Self> for TracingBackend<B> {
    fn sum_all<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::sum_all(&t.inner)?;
        Ok(Self::trace_unary(OpType::SumAll, t, &inner))
    }

    fn mean_all<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::mean_all(&t.inner)?;
        Ok(Self::trace_unary(OpType::MeanAll, t, &inner))
    }

    fn max_all<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::max_all(&t.inner)?;
        Ok(Self::trace_unary(OpType::MaxAll, t, &inner))
    }

    fn min_all<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::min_all(&t.inner)?;
        Ok(Self::trace_unary(OpType::MinAll, t, &inner))
    }

    fn sum_dim<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::sum_dim(&t.inner, dim)?;
        Ok(Self::trace_unary(OpType::SumDim, t, &inner))
    }

    fn sum_keepdim<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::sum_keepdim(&t.inner, dim)?;
        Ok(Self::trace_unary(OpType::SumDim, t, &inner))
    }

    fn mean_dim<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::mean_dim(&t.inner, dim)?;
        Ok(Self::trace_unary(OpType::MeanDim, t, &inner))
    }

    fn mean_keepdim<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::mean_keepdim(&t.inner, dim)?;
        Ok(Self::trace_unary(OpType::MeanDim, t, &inner))
    }

    fn max_dim<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::max_dim(&t.inner, dim)?;
        Ok(Self::trace_unary(OpType::MaxDim, t, &inner))
    }

    fn max_keepdim<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::max_keepdim(&t.inner, dim)?;
        Ok(Self::trace_unary(OpType::MaxDim, t, &inner))
    }

    fn min_dim<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::min_dim(&t.inner, dim)?;
        Ok(Self::trace_unary(OpType::MinDim, t, &inner))
    }

    fn min_keepdim<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::min_keepdim(&t.inner, dim)?;
        Ok(Self::trace_unary(OpType::MinDim, t, &inner))
    }

    fn argmax<K: super::dtype::DType, KInt: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: Option<usize>,
    ) -> Result<<Self as Backend>::Storage<KInt>> {
        let inner = B::argmax(&t.inner, dim)?;
        Ok(Self::trace_unary(OpType::ArgMax, t, &inner))
    }

    fn argmin<K: super::dtype::DType, KInt: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: Option<usize>,
    ) -> Result<<Self as Backend>::Storage<KInt>> {
        let inner = B::argmin(&t.inner, dim)?;
        Ok(Self::trace_unary(OpType::ArgMin, t, &inner))
    }
}

impl<B: Backend> TensorOps<Self> for TracingBackend<B> {
    fn tensor_to_dtype<K1: super::dtype::DType, K2: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K1>,
        dtype: KindleDType,
    ) -> Result<<Self as Backend>::Storage<K2>> {
        let inner = B::tensor_to_dtype::<K1, K2>(&t.inner, dtype)?;
        let shape = B::shape(&inner);
        let value_id = {
            let mut g = TRACING_GRAPH.lock();
            let out_id = g.add_value(shape, dtype, None);
            g.add_node(
                OpType::ToDtype,
                vec![t.value_id],
                vec![out_id],
                hashbrown::HashMap::new(),
            );
            out_id
        };
        Ok(TracingTensor { inner, value_id })
    }

    fn broadcast_as<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::broadcast_as(&t.inner, shape)?;
        Ok(Self::trace_unary(OpType::Broadcast, t, &inner))
    }

    fn reshape<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::reshape(&t.inner, shape)?;
        let shape_out = B::shape(&inner);
        let value_id = {
            let mut g = TRACING_GRAPH.lock();
            let out_id = g.add_value(shape_out.clone(), KindleDType::F32, None);

            // Add reshape parameters as a constant value
            let shape_val_id = g.add_value(vec![shape.len()], KindleDType::I64, None);
            let mut bytes = Vec::new();
            for &s in shape {
                bytes.extend_from_slice(&(s as i64).to_le_bytes());
            }
            g.initializers.insert(shape_val_id, bytes);

            g.add_node(
                OpType::Reshape,
                vec![t.value_id, shape_val_id],
                vec![out_id],
                hashbrown::HashMap::new(),
            );
            out_id
        };
        Ok(TracingTensor { inner, value_id })
    }

    fn transpose<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
        dim1: usize,
        dim2: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::transpose(&t.inner, dim1, dim2)?;
        let shape_out = B::shape(&inner);
        let value_id = {
            let mut g = TRACING_GRAPH.lock();
            let out_id = g.add_value(shape_out.clone(), KindleDType::F32, None);
            let mut attrs = hashbrown::HashMap::new();
            // simple perm vector building for ONNX
            let mut perm: Vec<i64> = (0..shape_out.len() as i64).collect();
            perm.swap(dim1, dim2);
            attrs.insert(alloc::string::String::from("perm"), crate::graph::AttributeValue::Ints(perm));
            g.add_node(OpType::Transpose, vec![t.value_id], vec![out_id], attrs);
            out_id
        };
        Ok(TracingTensor { inner, value_id })
    }

    fn narrow<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::narrow(&t.inner, dim, start, len)?;
        Ok(Self::trace_unary(OpType::Narrow, t, &inner))
    }

    fn concat<K: super::dtype::DType>(
        tensors: &[&<Self as Backend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inners: Vec<&B::Storage<K>> = tensors.iter().map(|t| &t.inner).collect();
        let inner = B::concat(&inners, dim)?;
        let shape_out = B::shape(&inner);
        let value_id = {
            let mut g = TRACING_GRAPH.lock();
            let out_id = g.add_value(shape_out, KindleDType::F32, None);
            let inputs = tensors.iter().map(|t| t.value_id).collect();
            let mut attrs = hashbrown::HashMap::new();
            attrs.insert(
                alloc::string::String::from("axis"),
                crate::graph::AttributeValue::Int(dim as i64),
            );
            g.add_node(OpType::Concat, inputs, vec![out_id], attrs);
            out_id
        };
        Ok(TracingTensor { inner, value_id })
    }

    fn stack<K: super::dtype::DType>(
        tensors: &[&<Self as Backend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inners: Vec<&B::Storage<K>> = tensors.iter().map(|t| &t.inner).collect();
        let inner = B::stack(&inners, dim)?;
        let shape_out = B::shape(&inner);
        let value_id = {
            let mut g = TRACING_GRAPH.lock();
            let out_id = g.add_value(shape_out, KindleDType::F32, None);
            let inputs = tensors.iter().map(|t| t.value_id).collect();
            let mut attrs = hashbrown::HashMap::new();
            attrs.insert(
                alloc::string::String::from("axis"),
                crate::graph::AttributeValue::Int(dim as i64),
            );
            g.add_node(OpType::Stack, inputs, vec![out_id], attrs);
            out_id
        };
        Ok(TracingTensor { inner, value_id })
    }

    fn slice<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
        ranges: &[(usize, usize)],
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::slice(&t.inner, ranges)?;
        Ok(Self::trace_unary(OpType::Slice, t, &inner))
    }

    fn flatten<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
        start_dim: usize,
        end_dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::flatten(&t.inner, start_dim, end_dim)?;
        Ok(Self::trace_unary(OpType::Reshape, t, &inner))
    }

    fn squeeze<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::squeeze(&t.inner, dim)?;
        Ok(Self::trace_unary(OpType::Reshape, t, &inner))
    }

    fn broadcast_left<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::broadcast_left(&t.inner, shape)?;
        Ok(Self::trace_unary(OpType::Broadcast, t, &inner))
    }

    fn matmul<K: super::dtype::DType>(
        lhs: &<Self as Backend>::Storage<K>,
        rhs: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::matmul(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(OpType::MatMul, lhs, rhs, &inner))
    }
    fn float_to_scalar<K: super::dtype::DType>(t: &<Self as Backend>::Storage<K>) -> Result<f64> {
        B::float_to_scalar(&t.inner)
    }
    fn float_to_vec1<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<alloc::vec::Vec<f64>> {
        B::float_to_vec1(&t.inner)
    }
    fn int_to_scalar<K: super::dtype::DType>(t: &<Self as Backend>::Storage<K>) -> Result<i64> {
        B::int_to_scalar(&t.inner)
    }
    fn int_to_vec1<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<alloc::vec::Vec<i64>> {
        B::int_to_vec1(&t.inner)
    }
}

impl<B: Backend + crate::tensor::backend::ModuleOps<B>> ModuleOps<Self> for TracingBackend<B> {
    fn conv1d<K: super::dtype::DType>(
        x: &<Self as Backend>::Storage<K>,
        weight: &<Self as Backend>::Storage<K>,
        bias: Option<&<Self as Backend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner_bias = bias.map(|b| &b.inner);
        let inner = B::conv1d(
            &x.inner,
            &weight.inner,
            inner_bias,
            stride,
            padding,
            dilation,
            groups,
        )?;
        let shape_out = B::shape(&inner);
        let value_id = {
            let mut g = TRACING_GRAPH.lock();
            let out_id = g.add_value(shape_out, KindleDType::F32, None);
            let mut inputs = vec![x.value_id, weight.value_id];
            if let Some(b) = bias {
                inputs.push(b.value_id);
            }
            let mut attrs = hashbrown::HashMap::new();
            attrs.insert(
                alloc::string::String::from("strides"),
                crate::graph::AttributeValue::Ints(vec![stride as i64]),
            );
            attrs.insert(
                alloc::string::String::from("pads"),
                crate::graph::AttributeValue::Ints(vec![padding as i64, padding as i64]),
            );
            attrs.insert(
                alloc::string::String::from("dilations"),
                crate::graph::AttributeValue::Ints(vec![dilation as i64]),
            );

            g.add_node(OpType::Conv1d, inputs, vec![out_id], attrs);
            out_id
        };
        Ok(TracingTensor { inner, value_id })
    }

    fn conv2d<K: super::dtype::DType>(
        x: &<Self as Backend>::Storage<K>,
        weight: &<Self as Backend>::Storage<K>,
        bias: Option<&<Self as Backend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner_bias = bias.map(|b| &b.inner);
        let inner = B::conv2d(
            &x.inner,
            &weight.inner,
            inner_bias,
            stride,
            padding,
            dilation,
            groups,
        )?;
        let shape_out = B::shape(&inner);
        let value_id = {
            let mut g = TRACING_GRAPH.lock();
            let out_id = g.add_value(shape_out, KindleDType::F32, None);
            let mut inputs = vec![x.value_id, weight.value_id];
            if let Some(b) = bias {
                inputs.push(b.value_id);
            }
            let mut attrs = hashbrown::HashMap::new();
            attrs.insert(
                alloc::string::String::from("strides"),
                crate::graph::AttributeValue::Ints(vec![stride as i64, stride as i64]),
            );
            attrs.insert(
                alloc::string::String::from("pads"),
                crate::graph::AttributeValue::Ints(vec![
                    padding as i64,
                    padding as i64,
                    padding as i64,
                    padding as i64,
                ]),
            );
            attrs.insert(
                alloc::string::String::from("dilations"),
                crate::graph::AttributeValue::Ints(vec![dilation as i64, dilation as i64]),
            );

            g.add_node(OpType::Conv2d, inputs, vec![out_id], attrs);
            out_id
        };
        Ok(TracingTensor { inner, value_id })
    }

    fn max_pool2d<K: super::dtype::DType>(
        x: &<Self as Backend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::max_pool2d(&x.inner, kernel_size, stride, padding, dilation)?;
        Ok(Self::trace_unary(OpType::MaxPool2d, x, &inner))
    }

    fn avg_pool2d<K: super::dtype::DType>(
        x: &<Self as Backend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::avg_pool2d(&x.inner, kernel_size, stride, padding)?;
        Ok(Self::trace_unary(OpType::AvgPool2d, x, &inner))
    }

    fn adaptive_avg_pool2d<K: super::dtype::DType>(
        x: &<Self as Backend>::Storage<K>,
        output_size: (usize, usize),
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::adaptive_avg_pool2d(&x.inner, output_size)?;
        Ok(Self::trace_unary(OpType::AdaptiveAvgPool2d, x, &inner))
    }

    fn embedding<K: super::dtype::DType, KInt: super::dtype::DType>(
        t: &<Self as Backend>::Storage<KInt>,
        w: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::embedding(&t.inner, &w.inner)?;
        let shape = B::shape(&inner);
        let value_id = {
            let mut g = TRACING_GRAPH.lock();
            let out_id = g.add_value(shape, KindleDType::F32, None);
            g.add_node(
                OpType::Embedding,
                vec![t.value_id, w.value_id],
                vec![out_id],
                hashbrown::HashMap::new(),
            );
            out_id
        };
        Ok(TracingTensor { inner, value_id })
    }

    fn layer_norm<K: super::dtype::DType>(
        x: &<Self as Backend>::Storage<K>,
        weight: &<Self as Backend>::Storage<K>,
        bias: Option<&<Self as Backend>::Storage<K>>,
        eps: f32,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::layer_norm(&x.inner, &weight.inner, bias.map(|b| &b.inner), eps)?;
        Ok(Self::trace_unary(OpType::LayerNorm, x, &inner))
    }

    fn batch_norm<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
        w: Option<&<Self as Backend>::Storage<K>>,
        b: Option<&<Self as Backend>::Storage<K>>,
        rm: Option<&<Self as Backend>::Storage<K>>,
        rv: Option<&<Self as Backend>::Storage<K>>,
        e: f32,
        momentum: f64,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::batch_norm(
            &t.inner,
            w.map(|x| &x.inner),
            b.map(|x| &x.inner),
            rm.map(|x| &x.inner),
            rv.map(|x| &x.inner),
            e,
            momentum,
        )?;
        Ok(Self::trace_unary(OpType::BatchNorm, t, &inner))
    }

    fn conv_transpose2d<K: super::dtype::DType>(
        t: &<Self as Backend>::Storage<K>,
        w: &<Self as Backend>::Storage<K>,
        b: Option<&<Self as Backend>::Storage<K>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner_b = b.map(|b| &b.inner);
        let inner = B::conv_transpose2d(
            &t.inner,
            &w.inner,
            inner_b,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        )?;
        Ok(Self::trace_unary(OpType::ConvTranspose2d, t, &inner))
    }
}

impl<B: Backend + crate::tensor::backend::LossOps<B>> LossOps<Self> for TracingBackend<B> {
    fn cross_entropy_loss<K: super::dtype::DType, KInt: super::dtype::DType>(
        logits: &<Self as Backend>::Storage<K>,
        targets: &<Self as Backend>::Storage<KInt>,
        reduction: crate::nn::loss::Reduction,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::cross_entropy_loss(&logits.inner, &targets.inner, reduction)?;
        Ok(Self::trace_binary(
            OpType::CrossEntropyLoss,
            logits,
            targets,
            &inner,
        ))
    }

    fn mse_loss<K: super::dtype::DType>(
        predictions: &<Self as Backend>::Storage<K>,
        targets: &<Self as Backend>::Storage<K>,
        reduction: crate::nn::loss::Reduction,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::mse_loss(&predictions.inner, &targets.inner, reduction)?;
        Ok(Self::trace_binary(
            OpType::MseLoss,
            predictions,
            targets,
            &inner,
        ))
    }

    fn l1_loss<K: super::dtype::DType>(
        predictions: &<Self as Backend>::Storage<K>,
        targets: &<Self as Backend>::Storage<K>,
        reduction: crate::nn::loss::Reduction,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::l1_loss(&predictions.inner, &targets.inner, reduction)?;
        Ok(Self::trace_binary(
            OpType::L1Loss,
            predictions,
            targets,
            &inner,
        ))
    }

    fn bce_with_logits_loss<K: super::dtype::DType>(
        logits: &<Self as Backend>::Storage<K>,
        targets: &<Self as Backend>::Storage<K>,
        _r: crate::nn::loss::Reduction,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::bce_with_logits_loss(&logits.inner, &targets.inner, _r)?;
        Ok(Self::trace_binary(
            OpType::BceWithLogitsLoss,
            logits,
            targets,
            &inner,
        ))
    }
}

impl<B: Backend> QuantizedOps<Self> for TracingBackend<B> {
    fn quantize<K: super::dtype::FloatDType, Q: super::dtype::QuantDType>(
        t: &<Self as Backend>::Storage<K>,
    ) -> Result<<Self as Backend>::Storage<Q>> {
        let inner = B::quantize(&t.inner)?;
        Ok(TracingTensor {
            inner,
            value_id: t.value_id, // simplistic trace
        })
    }

    fn dequantize<Q: super::dtype::QuantDType, K: super::dtype::FloatDType>(
        t: &<Self as Backend>::Storage<Q>,
    ) -> Result<<Self as Backend>::Storage<K>> {
        let inner = B::dequantize(&t.inner)?;
        Ok(TracingTensor {
            inner,
            value_id: t.value_id, // simplistic trace
        })
    }

    fn quantized_matmul<Q: super::dtype::QuantDType>(
        lhs: &<Self as Backend>::Storage<Q>,
        rhs: &<Self as Backend>::Storage<Q>,
    ) -> Result<<Self as Backend>::Storage<f32>> {
        let inner = B::quantized_matmul(&lhs.inner, &rhs.inner)?;
        Ok(TracingTensor {
            inner,
            value_id: lhs.value_id, // simplistic trace
        })
    }
}

impl<B: Backend> OptimizerOps<Self> for TracingBackend<B> {}
