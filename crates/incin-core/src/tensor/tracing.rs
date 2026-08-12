use crate::exec::catalog::TraceDescriptor;
use crate::exec::spec::ExecutionDescriptor;
use crate::graph::{Graph, ValueId};
use crate::prelude::*;
use crate::tensor::backend::*;
// removed RefCell
use spin::{Lazy, Mutex};

// Private per B-3 (.agents/API_DESIGN.md "pub(crate) is default"): this used
// to be `pub`, letting any downstream crate `.lock()` the raw `Mutex<Graph>`
// and call arbitrary `Graph` methods directly, even though `Graph` itself is
// `pub(crate)`. The three functions below are the only operations downstream
// crates actually need (draining/snapshotting the graph, marking an input/
// output value) — everything else about `Graph`'s shape stays encapsulated.
pub(crate) static TRACING_GRAPH: Lazy<Mutex<Graph>> = Lazy::new(|| Mutex::new(Graph::new()));

/// Drain the process-wide tracing graph, returning everything recorded since
/// the last call (or since startup).
pub fn extract_graph() -> Graph {
    let mut b = TRACING_GRAPH.lock();
    core::mem::take(&mut *b)
}

/// Clone the process-wide tracing graph's current contents WITHOUT draining
/// it (unlike `extract_graph`) — used by telemetry to snapshot mid-flight.
/// Returns `None` if the graph is momentarily locked elsewhere.
pub fn tracing_graph_snapshot() -> Option<Graph> {
    TRACING_GRAPH.try_lock().map(|g| (*g).clone())
}

/// Mark `value_id` as a graph input (e.g. for ONNX export / visualization).
pub fn tracing_mark_input(value_id: ValueId) {
    TRACING_GRAPH.lock().mark_input(value_id);
}

/// Mark `value_id` as a graph output (e.g. for ONNX export / visualization).
pub fn tracing_mark_output(value_id: ValueId) {
    TRACING_GRAPH.lock().mark_output(value_id);
}

/// A transparent wrapper around any `Backend` `B` that additionally
/// records every operation as a node in the process-wide `TRACING_GRAPH`,
/// building up an exportable computation graph (used by ONNX export and
/// graph visualization) alongside the real computation `B` performs.
use crate::exec::capability::Capabilities;

#[derive(Clone)]
pub struct TracingBackend<B: Backend> {
    _marker: core::marker::PhantomData<B>,
}

impl<B: Backend> core::default::Default for TracingBackend<B> {
    fn default() -> Self {
        Self {
            _marker: core::marker::PhantomData,
        }
    }
}

impl<B: Backend> Capabilities for TracingBackend<B> {
    fn support(&self, query: &crate::exec::CapabilityQuery) -> crate::exec::SupportLevel {
        B::default().support(query)
    }
}

#[derive(Clone)]
/// A `TracingBackend` storage handle: the real backend's storage plus the
/// `ValueId` identifying this tensor's node in the tracing graph.
pub struct TracingTensor<T> {
    /// The wrapped real backend storage.
    pub inner: T,
    /// This tensor's node id in the tracing graph.
    pub value_id: ValueId,
}

impl<T: crate::tensor::backend::ExecuteOutput> crate::tensor::backend::ExecuteOutput
    for TracingTensor<T>
{
}

#[derive(Clone)]
/// A `TracingBackend` variable handle: the real backend's `RawVar` plus
/// the `ValueId` identifying this variable's node in the tracing graph.
pub struct TracingVar<V> {
    /// The wrapped real backend variable.
    pub inner: V,
    /// This variable's node id in the tracing graph.
    pub value_id: ValueId,
}

impl<V> From<TracingTensor<V>> for TracingVar<V> {
    fn from(storage: TracingTensor<V>) -> Self {
        Self {
            inner: storage.inner,
            value_id: storage.value_id,
        }
    }
}

impl<B: Backend> TracingBackend<B> {
    fn traced_dtype<K: super::dtype::DType>(storage: &B::Storage<K>) -> DTypeId {
        B::storage_dtype(storage)
            .and_then(|dtype| dtype.builtin_id())
            .unwrap_or(DTypeId::F32)
    }

    /// Records a binary op's output as a new graph node with `lhs`/`rhs`
    /// as its inputs, wrapping `inner_res` with the new node's id.
    fn trace_binary<K1: super::dtype::DType, K2: super::dtype::DType, KOut: super::dtype::DType>(
        operation: OperationKind,
        lhs: &TracingTensor<B::Storage<K1>>,
        rhs: &TracingTensor<B::Storage<K2>>,
        inner_res: &B::Storage<KOut>,
    ) -> TracingTensor<B::Storage<KOut>> {
        let shape = B::shape(inner_res);
        let value_id = {
            let mut g = TRACING_GRAPH.lock();
            let out_id = g.add_value(shape.as_ref().to_vec(), Self::traced_dtype(inner_res), None);
            g.add_node(
                operation,
                vec![lhs.value_id, rhs.value_id],
                vec![out_id],
                alloc::collections::BTreeMap::new(),
            );
            out_id
        };
        TracingTensor {
            inner: inner_res.clone(),
            value_id,
        }
    }

    /// Records an op's output as a new graph node with an arbitrary number of
    /// already-traced inputs, for the ops that take three or more operands
    /// (`where_cond`, `scatter`, `addmm`, attention) and so fit neither
    /// [`Self::trace_unary`] nor [`Self::trace_binary`].
    fn trace_nary<KOut: super::dtype::DType>(
        operation: OperationKind,
        inputs: alloc::vec::Vec<ValueId>,
        inner_res: &B::Storage<KOut>,
    ) -> TracingTensor<B::Storage<KOut>> {
        let shape = B::shape(inner_res);
        let value_id = {
            let mut g = TRACING_GRAPH.lock();
            let out_id = g.add_value(shape.as_ref().to_vec(), Self::traced_dtype(inner_res), None);
            g.add_node(
                operation,
                inputs,
                vec![out_id],
                alloc::collections::BTreeMap::new(),
            );
            out_id
        };
        TracingTensor {
            inner: inner_res.clone(),
            value_id,
        }
    }

    /// Records a unary op's output as a new graph node with `t` as its
    /// input, wrapping `inner_res` with the new node's id.
    fn trace_unary<K: super::dtype::DType, KOut: super::dtype::DType>(
        operation: OperationKind,
        t: &TracingTensor<B::Storage<K>>,
        inner_res: &B::Storage<KOut>,
    ) -> TracingTensor<B::Storage<KOut>> {
        let shape = B::shape(inner_res);
        let value_id = {
            let mut g = TRACING_GRAPH.lock();
            let out_id = g.add_value(shape.as_ref().to_vec(), Self::traced_dtype(inner_res), None);
            g.add_node(
                operation,
                vec![t.value_id],
                vec![out_id],
                alloc::collections::BTreeMap::new(),
            );
            out_id
        };
        TracingTensor {
            inner: inner_res.clone(),
            value_id,
        }
    }
}

impl<B: Backend> crate::tensor::backend::StorageBackend for TracingBackend<B> {
    const BACKEND_NAME: &'static str = "tracing";
    /// The wrapped backend's storage plus a tracing-graph node id.
    type Storage<K: super::dtype::DType> = TracingTensor<B::Storage<K>>;
    /// Delegates to `B`'s device type.
    type Device = B::Device;

    fn metadata<K: super::dtype::DType>(storage: &Self::Storage<K>) -> &crate::exec::TensorMeta {
        B::metadata(&storage.inner)
    }

    fn fresh_autograd_identity<K: super::dtype::DType>(
        storage: Self::Storage<K>,
    ) -> Self::Storage<K> {
        TracingTensor {
            inner: B::fresh_autograd_identity(storage.inner),
            value_id: storage.value_id,
        }
    }

    fn execution_storage<K: super::dtype::DType>(
        storage: &Self::Storage<K>,
    ) -> (&dyn core::any::Any, Option<usize>)
    where
        Self::Storage<K>: core::any::Any,
        B::Storage<K>: core::any::Any,
    {
        (&storage.inner, Some(storage.value_id))
    }
}

impl<B: Backend + crate::tensor::backend::Execute<O>, O: crate::exec::catalog::Operation>
    crate::tensor::backend::Execute<O> for TracingBackend<B>
{
    type Output = TracingTensor<B::Output>;

    fn execute_shaped<S: crate::prelude::Shape>(
        &self,
        request: crate::tensor::backend::ExecutionRequest<'_, O, Self>,
    ) -> core::result::Result<Self::Output, BackendError> {
        let inner_backend = B::default();
        let inner_context = crate::exec::ExecutionContext::from_scope(inner_backend.clone());
        let inner_inputs: alloc::vec::Vec<_> = request
            .inputs
            .iter()
            .map(crate::exec::request::TensorHandle::execution_view)
            .collect();
        let inner_res =
            inner_backend.execute_shaped::<S>(crate::tensor::backend::ExecutionRequest {
                operation: request.operation,
                inputs: &inner_inputs,
                context: &inner_context,
                payload: request.payload,
            })?;

        let output_id = {
            let mut g = TRACING_GRAPH.lock();
            let output_id = g.add_value(
                request
                    .operation
                    .descriptor()
                    .output_shape()
                    .cloned()
                    .or_else(|| {
                        request
                            .inputs
                            .first()
                            .map(|input| input.metadata().shape.clone())
                    })
                    .unwrap_or(crate::shapes::ShapeBuf::SCALAR)
                    .as_ref()
                    .to_vec(),
                request
                    .operation
                    .descriptor()
                    .trace_output_dtype(request.inputs),
                None,
            );
            let inputs = request
                .inputs
                .iter()
                .filter_map(crate::exec::request::TensorHandle::tracing_value)
                .collect();
            g.add_node_with_identity(
                request.operation.descriptor().trace_identity(),
                inputs,
                vec![output_id],
                alloc::collections::BTreeMap::new(),
            );
            output_id
        };

        Ok(TracingTensor {
            inner: inner_res,
            value_id: output_id,
        })
    }
}

impl<B: Backend> Backend for TracingBackend<B> {
    /// The wrapped backend's variable plus a tracing-graph node id.
    type RawVar = TracingVar<B::RawVar>;
    /// Delegates to `B`'s gradient collection type — tracing adds no gradient bookkeeping of its own.
    type Grads = B::Grads;
    /// Delegates to `B`'s own inner backend (tracing is not itself a dispatch layer).
    type InnerBackend = B::InnerBackend;

    /// Delegates to the wrapped backend's canonical `ShapeBuf`.
    fn shape<K: super::dtype::DType>(
        t: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
    ) -> crate::shapes::ShapeBuf {
        B::shape(&t.inner)
    }

    fn storage_dtype<K: super::dtype::DType>(
        t: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
    ) -> Option<DTypeDescriptor> {
        B::storage_dtype(&t.inner)
    }

    fn storage_device<K: super::dtype::DType>(
        t: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
    ) -> Option<DeviceId> {
        B::storage_device(&t.inner)
    }

    fn format_tensor_display<K: super::dtype::DType>(
        t: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
    ) -> alloc::string::String
    where
        Self: crate::tensor::backend::TensorOps<Self>,
    {
        use crate::tensor::backend::TensorOps;
        use crate::tensor::display::{Values, render};
        let shape = Self::shape(t);
        match Self::storage_dtype(t) {
            None => alloc::format!("<tensor: shape={shape:?}, dtype unknown to this backend>"),
            Some(dtype) if dtype.is_quantized() => alloc::format!(
                "<{} tensor: shape={shape:?}, not printable without dequantizing>",
                dtype.name()
            ),
            Some(dtype) if dtype.is_integer() => match Self::int_to_vec1(t) {
                Ok(values) => render(&shape, &Values::Int(values)),
                Err(err) => alloc::format!("<tensor: shape={shape:?}, values unavailable: {err}>"),
            },
            Some(_) => match Self::float_to_vec1(t) {
                Ok(values) => render(&shape, &Values::Float(values)),
                Err(err) => alloc::format!("<tensor: shape={shape:?}, values unavailable: {err}>"),
            },
        }
    }
    /// Renders a tensor's values and metadata for `Debug`.
    fn format_tensor_debug<K: super::dtype::DType>(
        t: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
    ) -> alloc::string::String
    where
        Self: crate::tensor::backend::TensorOps<Self>,
    {
        Self::format_tensor_display::<K>(t)
    }

    /// Delegates to `B::var_as_tensor`, carrying the variable's tracing-graph node id over to the resulting storage.
    fn var_as_tensor<K: super::dtype::DType>(
        var: &<Self as Backend>::RawVar,
    ) -> Result<<Self as crate::tensor::backend::StorageBackend>::Storage<K>> {
        let inner = B::var_as_tensor(&var.inner)?;
        Ok(TracingTensor {
            inner,
            value_id: var.value_id,
        })
    }

    /// Delegates to `B::var_from_tensor`, carrying the storage's tracing-graph node id over to the resulting variable.
    fn var_from_tensor<K: super::dtype::DType>(
        t: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
    ) -> Result<<Self as Backend>::RawVar> {
        let inner = B::var_from_tensor(&t.inner)?;
        Ok(TracingVar {
            inner,
            value_id: t.value_id,
        })
    }

    /// Delegates to `B::assign_var`.
    fn assign_var<K: super::dtype::DType>(
        var: &mut <Self as Backend>::RawVar,
        tensor: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
    ) -> Result<()> {
        B::assign_var(&mut var.inner, &tensor.inner)
    }

    /// Delegates to `B::backward` — tracing does not itself affect gradient computation.
    fn backward<K: super::dtype::DType>(
        loss: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
    ) -> Result<<Self as Backend>::Grads> {
        B::backward(&loss.inner)
    }

    fn backward_with<K: super::dtype::DType>(
        loss: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
        seed: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
    ) -> Result<<Self as Backend>::Grads> {
        B::backward_with(&loss.inner, &seed.inner)
    }

    /// Always `None` — tracing doesn't maintain its own gradient map;
    /// gradient lookup must go through the wrapped backend `B` directly.
    fn get_grad<K: super::dtype::DType>(
        _var: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
        _grads: &<Self as Backend>::Grads,
    ) -> Result<Option<<Self as crate::tensor::backend::StorageBackend>::Storage<K>>> {
        Ok(None)
    }

    /// Delegates to `B::from_bytes`, additionally recording the result
    /// as a graph initializer (constant input) node.
    fn from_bytes<K: super::dtype::DType>(
        bytes: &[u8],
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as crate::tensor::backend::StorageBackend>::Storage<K>> {
        let inner = B::from_bytes(bytes, shape, dtype, device)?;
        let value_id = {
            let mut g = TRACING_GRAPH.lock();
            let id = g.add_value(
                shape.to_vec(),
                dtype.builtin_id().unwrap_or(DTypeId::F32),
                None,
            );
            g.initializers.insert(id, bytes.to_vec());
            id
        };
        Ok(TracingTensor { inner, value_id })
    }

    /// Delegates to `B::to_bytes`.
    fn to_bytes<K: super::dtype::DType>(
        t: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
    ) -> Result<alloc::vec::Vec<u8>> {
        B::to_bytes(&t.inner)
    }
}

impl<B: Backend + SupportsDType<K>, K: DType> SupportsDType<K> for TracingBackend<B> {
    fn resolve_dtype(field: &K::Field, device: &DeviceId) -> Result<DTypeDescriptor> {
        B::resolve_dtype(field, device)
    }
}

impl<B: Backend + CreationOps<B>> CreationOps<Self> for TracingBackend<B> {
    /// Delegates to `B::zeros`, additionally recording a new
    /// tracing-graph value node for the result.
    fn zeros<K: super::dtype::DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::zeros(shape, dtype, device)?;
        let value_id = TRACING_GRAPH.lock().add_value(
            shape.to_vec(),
            dtype.builtin_id().unwrap_or(DTypeId::F32),
            None,
        );
        Ok(TracingTensor { inner, value_id })
    }

    /// Delegates to `B::ones`, additionally recording a new
    /// tracing-graph value node for the result.
    fn ones<K: super::dtype::DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::ones(shape, dtype, device)?;
        let value_id = TRACING_GRAPH.lock().add_value(
            shape.to_vec(),
            dtype.builtin_id().unwrap_or(DTypeId::F32),
            None,
        );
        Ok(TracingTensor { inner, value_id })
    }

    /// Delegates to `B::full`, additionally recording a new
    /// tracing-graph value node for the result.
    fn full<K: super::dtype::DType>(
        val: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::full(val, shape, dtype, device)?;
        let value_id = TRACING_GRAPH.lock().add_value(
            shape.to_vec(),
            dtype.builtin_id().unwrap_or(DTypeId::F32),
            None,
        );
        Ok(TracingTensor { inner, value_id })
    }

    /// Delegates to `B::arange`, additionally recording a new
    /// tracing-graph value node for the result.
    fn arange<K: super::dtype::DType>(
        start: f64,
        step: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::arange(start, step, shape, dtype, device)?;
        let value_id = TRACING_GRAPH.lock().add_value(
            shape.to_vec(),
            dtype.builtin_id().unwrap_or(DTypeId::F32),
            None,
        );
        Ok(TracingTensor { inner, value_id })
    }

    /// Delegates to `B::linspace`, additionally recording a new
    /// tracing-graph value node for the result.
    fn linspace<K: super::dtype::DType>(
        start: f64,
        end: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::linspace(start, end, shape, dtype, device)?;
        let value_id = TRACING_GRAPH.lock().add_value(
            shape.to_vec(),
            dtype.builtin_id().unwrap_or(DTypeId::F32),
            None,
        );
        Ok(TracingTensor { inner, value_id })
    }

    /// Delegates to `B::rand`, additionally recording a new
    /// tracing-graph value node for the result.
    fn rand<K: super::dtype::DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::rand(shape, dtype, device)?;
        let value_id = TRACING_GRAPH.lock().add_value(
            shape.to_vec(),
            dtype.builtin_id().unwrap_or(DTypeId::F32),
            None,
        );
        Ok(TracingTensor { inner, value_id })
    }

    /// Delegates to `B::randn`, additionally recording a new
    /// tracing-graph value node for the result.
    fn randn<K: super::dtype::DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::randn(shape, dtype, device)?;
        let value_id = TRACING_GRAPH.lock().add_value(
            shape.to_vec(),
            dtype.builtin_id().unwrap_or(DTypeId::F32),
            None,
        );
        Ok(TracingTensor { inner, value_id })
    }

    /// Delegates to `B::var_zeros`, additionally recording a new
    /// tracing-graph value node for the result.
    fn var_zeros<K: super::dtype::DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::RawVar> {
        let inner = B::var_zeros::<K>(shape, dtype, device)?;
        let value_id = TRACING_GRAPH.lock().add_value(
            shape.to_vec(),
            dtype.builtin_id().unwrap_or(DTypeId::F32),
            None,
        );
        Ok(TracingVar { inner, value_id })
    }

    /// Delegates to `B::var_ones`, additionally recording a new
    /// tracing-graph value node for the result.
    fn var_ones<K: super::dtype::DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::RawVar> {
        let inner = B::var_ones::<K>(shape, dtype, device)?;
        let value_id = TRACING_GRAPH.lock().add_value(
            shape.to_vec(),
            dtype.builtin_id().unwrap_or(DTypeId::F32),
            None,
        );
        Ok(TracingVar { inner, value_id })
    }

    /// Delegates to `B::var_rand`, additionally recording a new
    /// tracing-graph value node for the result.
    fn var_rand<K: super::dtype::DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::RawVar> {
        let inner = B::var_rand::<K>(shape, dtype, device)?;
        let value_id = TRACING_GRAPH.lock().add_value(
            shape.to_vec(),
            dtype.builtin_id().unwrap_or(DTypeId::F32),
            None,
        );
        Ok(TracingVar { inner, value_id })
    }

    /// Delegates to `B::var_randn`, additionally recording a new
    /// tracing-graph value node for the result.
    fn var_randn<K: super::dtype::DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::RawVar> {
        let inner = B::var_randn::<K>(shape, dtype, device)?;
        let value_id = TRACING_GRAPH.lock().add_value(
            shape.to_vec(),
            dtype.builtin_id().unwrap_or(DTypeId::F32),
            None,
        );
        Ok(TracingVar { inner, value_id })
    }
}

impl<B, NewD> TransferTo<NewD> for TracingBackend<B>
where
    B: Backend + TransferTo<NewD>,
    NewD: crate::tensor::device::Device,
{
    type Output = TracingBackend<<B as TransferTo<NewD>>::Output>;

    fn transfer_storage<K: super::dtype::DType>(
        storage: &Self::Storage<K>,
        dtype: &K::Field,
        device: &NewD::Field,
    ) -> Result<<Self::Output as StorageBackend>::Storage<K>>
    where
        Self::Output: SupportsDType<K>,
    {
        let shape = B::shape(&storage.inner);
        let bytes = B::to_bytes(&storage.inner)?;
        let device_id = NewD::to_incin(device)?;
        let dtype_id = <Self::Output as SupportsDType<K>>::resolve_dtype(dtype, &device_id)?;
        let inner = <<B as TransferTo<NewD>>::Output as Backend>::from_bytes::<K>(
            &bytes, &shape, dtype_id, &device_id,
        )?;
        Ok(TracingTensor {
            inner,
            value_id: storage.value_id,
        })
    }

    fn transfer_var<K: DType>(
        variable: &Self::RawVar,
        dtype: &K::Field,
        device: &NewD::Field,
    ) -> Result<<Self::Output as Backend>::RawVar>
    where
        Self::Output: SupportsDType<K>,
    {
        let source = B::var_as_tensor::<K>(&variable.inner)?;
        let shape = B::shape(&source);
        let bytes = B::to_bytes(&source)?;
        let device_id = NewD::to_incin(device)?;
        let dtype_id = <Self::Output as SupportsDType<K>>::resolve_dtype(dtype, &device_id)?;
        let storage = <<B as TransferTo<NewD>>::Output as Backend>::from_bytes::<K>(
            &bytes, &shape, dtype_id, &device_id,
        )?;
        let inner = <<B as TransferTo<NewD>>::Output as Backend>::var_from_tensor(&storage)?;
        Ok(TracingVar {
            inner,
            value_id: variable.value_id,
        })
    }
}

impl<B: Backend + NumericOps<B>> NumericOps<Self> for TracingBackend<B> {
    /// Delegates to `B::add`, additionally recording an `OperationKind.Add` node.
    fn add<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::add(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(OperationKind::Add, lhs, rhs, &inner))
    }

    /// Delegates to `B::sub`, additionally recording an `OperationKind.Sub` node.
    fn sub<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::sub(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(OperationKind::Sub, lhs, rhs, &inner))
    }

    /// Delegates to `B::mul`, additionally recording an `OperationKind.Mul` node.
    fn mul<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::mul(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(OperationKind::Mul, lhs, rhs, &inner))
    }

    /// Delegates to `B::div`, additionally recording an `OperationKind.Div` node.
    fn div<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::div(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(OperationKind::Div, lhs, rhs, &inner))
    }
}

/// The error a tracing-only gap reports.
///
/// Identical to what `FloatOps`' removed default bodies produced, so this
/// changes where the refusal is written down, not what a caller observes.
fn untraceable<B: Backend>(op: &'static str) -> Error {
    Error::UnsupportedBackendOperation {
        op,
        backend: core::any::type_name::<TracingBackend<B>>(),
    }
}

/// Float operations the tracing graph has no node for.
///
/// [`OperationKind`] carries no variant for any of these, and ONNX export builds its
/// node list from that vocabulary, so recording them is not possible without
/// extending both. Delegating to `B` *without* recording would be worse than
/// refusing: the exported graph would silently omit the operation and stop
/// describing the model it came from.
///
/// Writing them out is the point. Until `EXE-009` these were inherited from
/// `FloatOps`' default bodies, which left an operation this backend cannot
/// trace indistinguishable, at this impl site, from one it can.
macro_rules! untraced_float_ops {
    (
        unary: $($unary:ident),* $(,)?;
        exponent: $($exponent:ident),* $(,)?;
        bounds: $($bounds:ident),* $(,)?;
        binary: $($binary:ident),* $(,)?;
    ) => {
        $(
            fn $unary<K: super::dtype::DType>(
                _t: &<Self as StorageBackend>::Storage<K>,
            ) -> Result<<Self as StorageBackend>::Storage<K>> {
                Err(untraceable::<B>(stringify!($unary)))
            }
        )*
        $(
            fn $exponent<K: super::dtype::DType>(
                _t: &<Self as StorageBackend>::Storage<K>,
                _exponent: f64,
            ) -> Result<<Self as StorageBackend>::Storage<K>> {
                Err(untraceable::<B>(stringify!($exponent)))
            }
        )*
        $(
            fn $bounds<K: super::dtype::DType>(
                _t: &<Self as StorageBackend>::Storage<K>,
                _min: f64,
                _max: f64,
            ) -> Result<<Self as StorageBackend>::Storage<K>> {
                Err(untraceable::<B>(stringify!($bounds)))
            }
        )*
        $(
            fn $binary<K: super::dtype::DType>(
                _lhs: &<Self as StorageBackend>::Storage<K>,
                _rhs: &<Self as StorageBackend>::Storage<K>,
            ) -> Result<<Self as StorageBackend>::Storage<K>> {
                Err(untraceable::<B>(stringify!($binary)))
            }
        )*
    };
}

impl<B: Backend + FloatOps<B>> FloatOps<Self> for TracingBackend<B> {
    untraced_float_ops! {
        unary: sign, floor, ceil, round, log2, log10, sin, cos, tan, asin, acos,
               atan, sinh, cosh, asinh, acosh, atanh, erf, rsqrt, trunc, frac;
        exponent: powf;
        bounds: clamp;
        binary: atan2, fmod, remainder;
    }

    /// Delegates to `B::add_scalar_float`, additionally recording an
    /// `OperationKind::AddScalar` node.
    fn add_scalar_float<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::add_scalar_float(&t.inner, scalar)?;
        Ok(Self::trace_unary(OperationKind::AddScalar, t, &inner))
    }

    /// Delegates to `B::mul_scalar_float`, additionally recording an
    /// `OperationKind::MulScalar` node.
    fn mul_scalar_float<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::mul_scalar_float(&t.inner, scalar)?;
        Ok(Self::trace_unary(OperationKind::MulScalar, t, &inner))
    }

    /// Delegates to `B::relu`, additionally recording an
    /// `OperationKind::Relu` node.
    fn relu<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::relu(&t.inner)?;
        Ok(Self::trace_unary(OperationKind::Relu, t, &inner))
    }

    fn step<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::step(&t.inner)?;
        Ok(Self::trace_unary(OperationKind::Step, t, &inner))
    }

    fn mish<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::mish(&t.inner)?;
        Ok(Self::trace_unary(OperationKind::Mish, t, &inner))
    }

    fn elu<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::elu(&t.inner)?;
        Ok(Self::trace_unary(OperationKind::Elu, t, &inner))
    }

    /// Delegates to `B::gelu`, additionally recording an
    /// `OperationKind::Gelu` node.
    fn gelu<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::gelu(&t.inner)?;
        Ok(Self::trace_unary(OperationKind::Gelu, t, &inner))
    }

    /// Delegates to `B::abs`, additionally recording an
    /// `OperationKind::Abs` node.
    fn abs<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::abs(&t.inner)?;
        Ok(Self::trace_unary(OperationKind::Abs, t, &inner))
    }

    /// Delegates to `B::exp`, additionally recording an
    /// `OperationKind::Exp` node.
    fn exp<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::exp(&t.inner)?;
        Ok(Self::trace_unary(OperationKind::Exp, t, &inner))
    }

    /// Delegates to `B::neg`, additionally recording an
    /// `OperationKind::Neg` node.
    fn neg<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::neg(&t.inner)?;
        Ok(Self::trace_unary(OperationKind::Neg, t, &inner))
    }

    /// Delegates to `B::sqrt`, additionally recording an
    /// `OperationKind::Sqrt` node.
    fn sqrt<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::sqrt(&t.inner)?;
        Ok(Self::trace_unary(OperationKind::Sqrt, t, &inner))
    }

    /// Delegates to `B::log`, additionally recording an
    /// `OperationKind::Log` node.
    fn log<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::log(&t.inner)?;
        Ok(Self::trace_unary(OperationKind::Log, t, &inner))
    }

    /// Delegates to `B::tanh`, additionally recording an
    /// `OperationKind::Tanh` node.
    fn tanh<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::tanh(&t.inner)?;
        Ok(Self::trace_unary(OperationKind::Tanh, t, &inner))
    }

    /// Delegates to `B::sigmoid`, additionally recording an
    /// `OperationKind::Sigmoid` node.
    fn sigmoid<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::sigmoid(&t.inner)?;
        Ok(Self::trace_unary(OperationKind::Sigmoid, t, &inner))
    }

    /// Delegates to `B::swish`, additionally recording an
    /// `OperationKind::Swish` node.
    fn swish<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::swish(&t.inner)?;
        Ok(Self::trace_unary(OperationKind::Swish, t, &inner))
    }

    /// Delegates to `B::softmax`, additionally recording an
    /// `OperationKind::Softmax` node with the reduced `dim` as an attribute.
    fn softmax<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::softmax(&t.inner, dim)?;
        let shape = B::shape(&inner);
        let value_id = {
            let mut g = TRACING_GRAPH.lock();
            let out_id = g.add_value(shape.as_ref().to_vec(), DTypeId::F32, None);
            let mut attrs = alloc::collections::BTreeMap::new();
            attrs.insert(
                alloc::string::String::from("axis"),
                crate::graph::AttributeValue::Int(dim as i64),
            );
            g.add_node(
                OperationKind::Softmax,
                vec![t.value_id],
                vec![out_id],
                attrs,
            );
            out_id
        };
        Ok(TracingTensor { inner, value_id })
    }
}

impl<B: Backend + ReductionOps<B>> ReductionOps<Self> for TracingBackend<B> {
    /// `OperationKind` has no product or cumulative-sum node, so these cannot be
    /// recorded. Refusing keeps an exported graph honest; delegating silently
    /// would drop the operation from the model the graph claims to describe.
    fn prod_all<K: super::dtype::DType>(
        _t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Err(untraceable::<B>("prod_all"))
    }

    /// Untraceable for the same reason as [`prod_all`](Self::prod_all).
    fn prod_dim<K: super::dtype::DType>(
        _t: &<Self as StorageBackend>::Storage<K>,
        _dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Err(untraceable::<B>("prod_dim"))
    }

    /// Untraceable for the same reason as [`prod_all`](Self::prod_all).
    fn cumsum<K: super::dtype::DType>(
        _t: &<Self as StorageBackend>::Storage<K>,
        _dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Err(untraceable::<B>("cumsum"))
    }

    /// Delegates to `B::sum_all`, additionally recording an `OperationKind::SumAll` node.
    fn sum_all<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::sum_all(&t.inner)?;
        Ok(Self::trace_unary(OperationKind::SumAll, t, &inner))
    }

    /// Delegates to `B::mean_all`, additionally recording an `OperationKind::MeanAll` node.
    fn mean_all<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::mean_all(&t.inner)?;
        Ok(Self::trace_unary(OperationKind::MeanAll, t, &inner))
    }

    /// Delegates to `B::max_all`, additionally recording an `OperationKind::MaxAll` node.
    fn max_all<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::max_all(&t.inner)?;
        Ok(Self::trace_unary(OperationKind::MaxAll, t, &inner))
    }

    /// Delegates to `B::min_all`, additionally recording an `OperationKind::MinAll` node.
    fn min_all<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::min_all(&t.inner)?;
        Ok(Self::trace_unary(OperationKind::MinAll, t, &inner))
    }

    /// Delegates to `B::sum_dim`, additionally recording an `OperationKind::SumDim` node.
    fn sum_dim<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::sum_dim(&t.inner, dim)?;
        Ok(Self::trace_unary(OperationKind::SumDim, t, &inner))
    }

    /// Delegates to `B::sum_keepdim`, additionally recording an `OperationKind::SumDim` node.
    fn sum_keepdim<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::sum_keepdim(&t.inner, dim)?;
        Ok(Self::trace_unary(OperationKind::SumDim, t, &inner))
    }

    /// Delegates to `B::mean_dim`, additionally recording an `OperationKind::MeanDim` node.
    fn mean_dim<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::mean_dim(&t.inner, dim)?;
        Ok(Self::trace_unary(OperationKind::MeanDim, t, &inner))
    }

    /// Delegates to `B::mean_keepdim`, additionally recording an `OperationKind::MeanDim` node.
    fn mean_keepdim<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::mean_keepdim(&t.inner, dim)?;
        Ok(Self::trace_unary(OperationKind::MeanDim, t, &inner))
    }

    /// Delegates to `B::max_dim`, additionally recording an `OperationKind::MaxDim` node.
    fn max_dim<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::max_dim(&t.inner, dim)?;
        Ok(Self::trace_unary(OperationKind::MaxDim, t, &inner))
    }

    /// Delegates to `B::max_keepdim`, additionally recording an `OperationKind::MaxDim` node.
    fn max_keepdim<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::max_keepdim(&t.inner, dim)?;
        Ok(Self::trace_unary(OperationKind::MaxDim, t, &inner))
    }

    /// Delegates to `B::min_dim`, additionally recording an `OperationKind::MinDim` node.
    fn min_dim<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::min_dim(&t.inner, dim)?;
        Ok(Self::trace_unary(OperationKind::MinDim, t, &inner))
    }

    /// Delegates to `B::min_keepdim`, additionally recording an `OperationKind::MinDim` node.
    fn min_keepdim<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::min_keepdim(&t.inner, dim)?;
        Ok(Self::trace_unary(OperationKind::MinDim, t, &inner))
    }

    /// Delegates to `B::argmax`, additionally recording an `OperationKind::ArgMax` node.
    fn argmax<K: super::dtype::DType, KInt: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: Option<usize>,
    ) -> Result<<Self as StorageBackend>::Storage<KInt>> {
        let inner = B::argmax(&t.inner, dim)?;
        Ok(Self::trace_unary(OperationKind::ArgMax, t, &inner))
    }

    /// Delegates to `B::argmin`, additionally recording an `OperationKind::ArgMin` node.
    fn argmin<K: super::dtype::DType, KInt: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: Option<usize>,
    ) -> Result<<Self as StorageBackend>::Storage<KInt>> {
        let inner = B::argmin(&t.inner, dim)?;
        Ok(Self::trace_unary(OperationKind::ArgMin, t, &inner))
    }

    /// Delegates to `B::topk`, recording both outputs (values and indices)
    /// in the tracing graph under `OperationKind::TopK` with `k`, `dim`, and
    /// `largest` stored as node attributes.
    fn topk<K: super::dtype::DType, KInt: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        k: usize,
        dim: usize,
        largest: bool,
    ) -> Result<(
        <Self as StorageBackend>::Storage<K>,
        <Self as StorageBackend>::Storage<KInt>,
    )> {
        let (v_inner, i_inner) = B::topk(&t.inner, k, dim, largest)?;

        let shape_v = B::shape(&v_inner);
        let shape_i = B::shape(&i_inner);

        let v_id = {
            let mut g = TRACING_GRAPH.lock();
            let out_id = g.add_value(shape_v.as_ref().to_vec(), DTypeId::F32, None);
            let mut attrs = alloc::collections::BTreeMap::new();
            attrs.insert(
                alloc::string::String::from("k"),
                crate::graph::AttributeValue::Int(k as i64),
            );
            attrs.insert(
                alloc::string::String::from("axis"),
                crate::graph::AttributeValue::Int(dim as i64),
            );
            attrs.insert(
                alloc::string::String::from("largest"),
                crate::graph::AttributeValue::Int(if largest { 1 } else { 0 }),
            );
            g.add_node(OperationKind::TopK, vec![t.value_id], vec![out_id], attrs);
            out_id
        };

        let i_id = {
            let mut g = TRACING_GRAPH.lock();
            let out_id = g.add_value(shape_i.as_ref().to_vec(), DTypeId::U32, None);
            let mut attrs = alloc::collections::BTreeMap::new();
            attrs.insert(
                alloc::string::String::from("k"),
                crate::graph::AttributeValue::Int(k as i64),
            );
            attrs.insert(
                alloc::string::String::from("axis"),
                crate::graph::AttributeValue::Int(dim as i64),
            );
            attrs.insert(
                alloc::string::String::from("largest"),
                crate::graph::AttributeValue::Int(if largest { 1 } else { 0 }),
            );
            g.add_node(OperationKind::TopK, vec![t.value_id], vec![out_id], attrs);
            out_id
        };

        Ok((
            TracingTensor {
                inner: v_inner,
                value_id: v_id,
            },
            TracingTensor {
                inner: i_inner,
                value_id: i_id,
            },
        ))
    }

    /// Delegates to `B::argsort`, recording an `OperationKind::Argsort` node
    /// with `dim` and `descending` as node attributes.
    fn argsort<K: super::dtype::DType, KInt: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        descending: bool,
    ) -> Result<<Self as StorageBackend>::Storage<KInt>> {
        let inner = B::argsort(&t.inner, dim, descending)?;
        let shape_out = B::shape(&inner);
        let value_id = {
            let mut g = TRACING_GRAPH.lock();
            let out_id = g.add_value(shape_out.as_ref().to_vec(), DTypeId::I64, None);
            let mut attrs = alloc::collections::BTreeMap::new();
            attrs.insert(
                alloc::string::String::from("axis"),
                crate::graph::AttributeValue::Int(dim as i64),
            );
            attrs.insert(
                alloc::string::String::from("descending"),
                crate::graph::AttributeValue::Int(if descending { 1 } else { 0 }),
            );
            g.add_node(
                OperationKind::Argsort,
                vec![t.value_id],
                vec![out_id],
                attrs,
            );
            out_id
        };
        Ok(TracingTensor { inner, value_id })
    }
}

impl<B: Backend + TensorOps<B>> TensorOps<Self> for TracingBackend<B> {
    /// Delegates to `B::tensor_to_dtype`, additionally recording an `OperationKind::ToDtype` node.
    fn tensor_to_dtype<K1: super::dtype::DType, K2: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K1>,
        dtype: DTypeDescriptor,
    ) -> Result<<Self as StorageBackend>::Storage<K2>> {
        let inner = B::tensor_to_dtype::<K1, K2>(&t.inner, dtype)?;
        let shape = B::shape(&inner);
        let value_id = {
            let mut g = TRACING_GRAPH.lock();
            let out_id = g.add_value(
                shape.as_ref().to_vec(),
                dtype.builtin_id().unwrap_or(DTypeId::F32),
                None,
            );
            g.add_node(
                OperationKind::ToDType,
                vec![t.value_id],
                vec![out_id],
                alloc::collections::BTreeMap::new(),
            );
            out_id
        };
        Ok(TracingTensor { inner, value_id })
    }

    /// Delegates to `B::broadcast_as`, additionally recording an `OperationKind::Broadcast` node.
    fn broadcast_as<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::broadcast_as(&t.inner, shape)?;
        Ok(Self::trace_unary(OperationKind::Broadcast, t, &inner))
    }

    /// Delegates to `B::reshape`, additionally recording an
    /// `OperationKind::Reshape` node with the target shape stored as a constant
    /// initializer input (matching ONNX's `Reshape` operator signature).
    fn reshape<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::reshape(&t.inner, shape)?;
        let shape_out = B::shape(&inner);
        let value_id = {
            let mut g = TRACING_GRAPH.lock();
            let out_id = g.add_value(shape_out.as_ref().to_vec(), DTypeId::F32, None);

            // Add reshape parameters as a constant value
            let shape_val_id = g.add_value(vec![shape.len()], DTypeId::I64, None);
            let mut bytes = Vec::new();
            for &s in shape {
                bytes.extend_from_slice(&(s as i64).to_le_bytes());
            }
            g.initializers.insert(shape_val_id, bytes);

            g.add_node(
                OperationKind::Reshape,
                vec![t.value_id, shape_val_id],
                vec![out_id],
                alloc::collections::BTreeMap::new(),
            );
            out_id
        };
        Ok(TracingTensor { inner, value_id })
    }

    /// Delegates to `B::transpose`, additionally recording an
    /// `OperationKind::Transpose` node with the resulting permutation as a `perm` attribute.
    fn transpose<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim1: usize,
        dim2: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::transpose(&t.inner, dim1, dim2)?;
        let shape_out = B::shape(&inner);
        let value_id = {
            let mut g = TRACING_GRAPH.lock();
            let out_id = g.add_value(shape_out.as_ref().to_vec(), DTypeId::F32, None);
            let mut attrs = alloc::collections::BTreeMap::new();
            // simple perm vector building for ONNX
            let mut perm: Vec<i64> = (0..shape_out.len() as i64).collect();
            perm.swap(dim1, dim2);
            attrs.insert(
                alloc::string::String::from("perm"),
                crate::graph::AttributeValue::Ints(perm),
            );
            g.add_node(
                OperationKind::Transpose,
                vec![t.value_id],
                vec![out_id],
                attrs,
            );
            out_id
        };
        Ok(TracingTensor { inner, value_id })
    }

    /// Delegates to `B::narrow`, additionally recording an `OperationKind::Narrow` node.
    fn narrow<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::narrow(&t.inner, dim, start, len)?;
        Ok(Self::trace_unary(OperationKind::Narrow, t, &inner))
    }

    /// Delegates to `B::concat`, additionally recording an
    /// `OperationKind::Concat` node with `dim` as an `axis` attribute.
    fn concat<K: super::dtype::DType>(
        tensors: &[&<Self as StorageBackend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inners: Vec<&B::Storage<K>> = tensors.iter().map(|t| &t.inner).collect();
        let inner = B::concat(&inners, dim)?;
        let shape_out = B::shape(&inner);
        let value_id = {
            let mut g = TRACING_GRAPH.lock();
            let out_id = g.add_value(shape_out.as_ref().to_vec(), DTypeId::F32, None);
            let inputs = tensors.iter().map(|t| t.value_id).collect();
            let mut attrs = alloc::collections::BTreeMap::new();
            attrs.insert(
                alloc::string::String::from("axis"),
                crate::graph::AttributeValue::Int(dim as i64),
            );
            g.add_node(OperationKind::Concat, inputs, vec![out_id], attrs);
            out_id
        };
        Ok(TracingTensor { inner, value_id })
    }

    /// Delegates to `B::stack`, additionally recording an
    /// `OperationKind::Stack` node with `dim` as an `axis` attribute.
    fn stack<K: super::dtype::DType>(
        tensors: &[&<Self as StorageBackend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inners: Vec<&B::Storage<K>> = tensors.iter().map(|t| &t.inner).collect();
        let inner = B::stack(&inners, dim)?;
        let shape_out = B::shape(&inner);
        let value_id = {
            let mut g = TRACING_GRAPH.lock();
            let out_id = g.add_value(shape_out.as_ref().to_vec(), DTypeId::F32, None);
            let inputs = tensors.iter().map(|t| t.value_id).collect();
            let mut attrs = alloc::collections::BTreeMap::new();
            attrs.insert(
                alloc::string::String::from("axis"),
                crate::graph::AttributeValue::Int(dim as i64),
            );
            g.add_node(OperationKind::Stack, inputs, vec![out_id], attrs);
            out_id
        };
        Ok(TracingTensor { inner, value_id })
    }

    /// Delegates to `B::slice`, additionally recording an `OperationKind::Slice` node.
    fn slice<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        ranges: &[(usize, usize)],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::slice(&t.inner, ranges)?;
        Ok(Self::trace_unary(OperationKind::Slice, t, &inner))
    }

    /// Delegates to `B::flatten`, additionally recording a
    /// (shape-only) `OperationKind::Reshape` node.
    fn flatten<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        start_dim: usize,
        end_dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::flatten(&t.inner, start_dim, end_dim)?;
        Ok(Self::trace_unary(OperationKind::Reshape, t, &inner))
    }

    /// Delegates to `B::squeeze`, additionally recording a
    /// (shape-only) `OperationKind::Reshape` node.
    fn squeeze<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::squeeze(&t.inner, dim)?;
        Ok(Self::trace_unary(OperationKind::Reshape, t, &inner))
    }

    /// Delegates to `B::broadcast_left`, additionally recording an `OperationKind::Broadcast` node.
    fn broadcast_left<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::broadcast_left(&t.inner, shape)?;
        Ok(Self::trace_unary(OperationKind::Broadcast, t, &inner))
    }

    /// Delegates to `B::matmul`, additionally recording an `OperationKind::MatMul` node.
    fn matmul<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::matmul(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(OperationKind::MatMul, lhs, rhs, &inner))
    }
    /// Delegates to `B::float_to_scalar`.
    fn float_to_scalar<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<f64> {
        B::float_to_scalar(&t.inner)
    }
    /// Delegates to `B::float_to_vec1`.
    fn float_to_vec1<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<alloc::vec::Vec<f64>> {
        B::float_to_vec1(&t.inner)
    }
    /// Delegates to `B::int_to_scalar`.
    fn int_to_scalar<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<i64> {
        B::int_to_scalar(&t.inner)
    }
    /// Delegates to `B::int_to_vec1`.
    fn int_to_vec1<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<alloc::vec::Vec<i64>> {
        B::int_to_vec1(&t.inner)
    }

    /// Delegates to `B::where_cond`, recording an `OperationKind::WhereCond` node
    /// whose inputs are the mask and both branches.
    fn where_cond<K: super::dtype::DType>(
        mask: &<Self as StorageBackend>::Storage<bool>,
        on_true: &<Self as StorageBackend>::Storage<K>,
        on_false: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::where_cond::<K>(&mask.inner, &on_true.inner, &on_false.inner)?;
        Ok(Self::trace_nary(
            OperationKind::WhereCond,
            alloc::vec![mask.value_id, on_true.value_id, on_false.value_id],
            &inner,
        ))
    }

    /// Delegates to `B::gather`, recording an `OperationKind::Gather` node.
    fn gather<K: super::dtype::DType, KInt: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        index: &<Self as StorageBackend>::Storage<KInt>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::gather::<K, KInt>(&t.inner, dim, &index.inner)?;
        Ok(Self::trace_nary(
            OperationKind::Gather,
            alloc::vec![t.value_id, index.value_id],
            &inner,
        ))
    }

    /// Delegates to `B::scatter`, recording an `OperationKind::Scatter` node whose
    /// inputs are the target, the index, and the source.
    fn scatter<K: super::dtype::DType, KInt: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        index: &<Self as StorageBackend>::Storage<KInt>,
        src: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::scatter::<K, KInt>(&t.inner, dim, &index.inner, &src.inner)?;
        Ok(Self::trace_nary(
            OperationKind::Scatter,
            alloc::vec![t.value_id, index.value_id, src.value_id],
            &inner,
        ))
    }

    /// Delegates to `B::index_select`, recording an `OperationKind::IndexSelect` node.
    fn index_select<K: super::dtype::DType, KInt: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        index: &<Self as StorageBackend>::Storage<KInt>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::index_select::<K, KInt>(&t.inner, dim, &index.inner)?;
        Ok(Self::trace_nary(
            OperationKind::IndexSelect,
            alloc::vec![t.value_id, index.value_id],
            &inner,
        ))
    }

    /// Delegates to `B::masked_fill`, recording an `OperationKind::MaskedFill` node.
    fn masked_fill<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        mask: &<Self as StorageBackend>::Storage<bool>,
        value: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::masked_fill::<K>(&t.inner, &mask.inner, value)?;
        Ok(Self::trace_nary(
            OperationKind::MaskedFill,
            alloc::vec![t.value_id, mask.value_id],
            &inner,
        ))
    }

    /// Delegates to `B::unsqueeze`, recording an `OperationKind::Unsqueeze` node.
    fn unsqueeze<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::unsqueeze(&t.inner, dim)?;
        Ok(Self::trace_unary(OperationKind::Unsqueeze, t, &inner))
    }

    /// Delegates to `B::repeat`, recording an `OperationKind::Repeat` node.
    fn repeat<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        repeats: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::repeat(&t.inner, repeats)?;
        Ok(Self::trace_unary(OperationKind::Repeat, t, &inner))
    }

    /// Delegates to `B::pad`, recording an `OperationKind::Pad` node.
    fn pad<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        padding: &[(usize, usize)],
        val: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::pad(&t.inner, padding, val)?;
        Ok(Self::trace_unary(OperationKind::Pad, t, &inner))
    }

    /// Delegates to `B::triu`, recording an `OperationKind::Triu` node.
    fn triu<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        k: i64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::triu(&t.inner, k)?;
        Ok(Self::trace_unary(OperationKind::Triu, t, &inner))
    }

    /// Delegates to `B::tril`, recording an `OperationKind::Tril` node.
    fn tril<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        k: i64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::tril(&t.inner, k)?;
        Ok(Self::trace_unary(OperationKind::Tril, t, &inner))
    }

    /// Delegates to `B::diag`, recording an `OperationKind::Diag` node.
    fn diag<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        k: i64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::diag(&t.inner, k)?;
        Ok(Self::trace_unary(OperationKind::Diag, t, &inner))
    }

    /// Delegates to `B::cmp_eq`, recording an `OperationKind::CmpEq` node.
    fn cmp_eq<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        let inner = B::cmp_eq(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(OperationKind::CmpEq, lhs, rhs, &inner))
    }

    /// Delegates to `B::cmp_ne`, recording an `OperationKind::CmpNe` node.
    fn cmp_ne<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        let inner = B::cmp_ne(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(OperationKind::CmpNe, lhs, rhs, &inner))
    }

    /// Delegates to `B::cmp_lt`, recording an `OperationKind::CmpLt` node.
    fn cmp_lt<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        let inner = B::cmp_lt(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(OperationKind::CmpLt, lhs, rhs, &inner))
    }

    /// Delegates to `B::cmp_le`, recording an `OperationKind::CmpLe` node.
    fn cmp_le<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        let inner = B::cmp_le(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(OperationKind::CmpLe, lhs, rhs, &inner))
    }

    /// Delegates to `B::cmp_gt`, recording an `OperationKind::CmpGt` node.
    fn cmp_gt<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        let inner = B::cmp_gt(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(OperationKind::CmpGt, lhs, rhs, &inner))
    }

    /// Delegates to `B::cmp_ge`, recording an `OperationKind::CmpGe` node.
    fn cmp_ge<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        let inner = B::cmp_ge(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(OperationKind::CmpGe, lhs, rhs, &inner))
    }

    /// Delegates to `B::logical_and`, recording an `OperationKind::LogicalAnd` node.
    fn logical_and(
        lhs: &<Self as StorageBackend>::Storage<bool>,
        rhs: &<Self as StorageBackend>::Storage<bool>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        let inner = B::logical_and(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(
            OperationKind::LogicalAnd,
            lhs,
            rhs,
            &inner,
        ))
    }

    /// Delegates to `B::logical_or`, recording an `OperationKind::LogicalOr` node.
    fn logical_or(
        lhs: &<Self as StorageBackend>::Storage<bool>,
        rhs: &<Self as StorageBackend>::Storage<bool>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        let inner = B::logical_or(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(
            OperationKind::LogicalOr,
            lhs,
            rhs,
            &inner,
        ))
    }

    /// Delegates to `B::logical_not`, recording an `OperationKind::LogicalNot` node.
    fn logical_not(
        t: &<Self as StorageBackend>::Storage<bool>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        let inner = B::logical_not(&t.inner)?;
        Ok(Self::trace_unary(OperationKind::LogicalNot, t, &inner))
    }

    /// Delegates to `B::sub_scalar`, recording an `OperationKind::SubScalar` node.
    fn sub_scalar<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        val: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::sub_scalar(&t.inner, val)?;
        Ok(Self::trace_unary(OperationKind::SubScalar, t, &inner))
    }

    /// Delegates to `B::div_scalar`, recording an `OperationKind::DivScalar` node.
    fn div_scalar<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        val: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::div_scalar(&t.inner, val)?;
        Ok(Self::trace_unary(OperationKind::DivScalar, t, &inner))
    }

    /// Delegates to `B::maximum`, recording an `OperationKind::Maximum` node.
    fn maximum<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::maximum(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(OperationKind::Maximum, lhs, rhs, &inner))
    }

    /// Delegates to `B::minimum`, recording an `OperationKind::Minimum` node.
    fn minimum<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::minimum(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(OperationKind::Minimum, lhs, rhs, &inner))
    }

    /// Delegates to `B::abs_diff`, recording an `OperationKind::AbsDiff` node.
    fn abs_diff<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::abs_diff(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(OperationKind::AbsDiff, lhs, rhs, &inner))
    }

    /// Delegates to `B::lerp`, recording an `OperationKind::Lerp` node.
    fn lerp<K: super::dtype::DType>(
        start: &<Self as StorageBackend>::Storage<K>,
        end: &<Self as StorageBackend>::Storage<K>,
        weight: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::lerp(&start.inner, &end.inner, weight)?;
        Ok(Self::trace_binary(OperationKind::Lerp, start, end, &inner))
    }

    /// Delegates to `B::addmm`, recording an `OperationKind::Addmm` node whose inputs
    /// are the added matrix and both multiplicands.
    fn addmm<K: super::dtype::DType>(
        mat: &<Self as StorageBackend>::Storage<K>,
        mat1: &<Self as StorageBackend>::Storage<K>,
        mat2: &<Self as StorageBackend>::Storage<K>,
        beta: f64,
        alpha: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::addmm(&mat.inner, &mat1.inner, &mat2.inner, beta, alpha)?;
        Ok(Self::trace_nary(
            OperationKind::Addmm,
            alloc::vec![mat.value_id, mat1.value_id, mat2.value_id],
            &inner,
        ))
    }

    /// Delegates to `B::bmm`, recording an `OperationKind::Bmm` node.
    fn bmm<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::bmm(&lhs.inner, &rhs.inner)?;
        Ok(Self::trace_binary(
            OperationKind::BatchedMatMul,
            lhs,
            rhs,
            &inner,
        ))
    }

    /// Delegates to `B::scaled_dot_product_attention`, recording an
    /// `OperationKind::ScaledDotProductAttention` node. The optional mask becomes a
    /// fourth input only when one was supplied, so the recorded arity matches
    /// the call.
    fn scaled_dot_product_attention<K: super::dtype::DType>(
        q: &<Self as StorageBackend>::Storage<K>,
        k: &<Self as StorageBackend>::Storage<K>,
        v: &<Self as StorageBackend>::Storage<K>,
        mask: Option<&<Self as StorageBackend>::Storage<K>>,
        scale: Option<f64>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::scaled_dot_product_attention(
            &q.inner,
            &k.inner,
            &v.inner,
            mask.map(|m| &m.inner),
            scale,
        )?;
        let mut inputs = alloc::vec![q.value_id, k.value_id, v.value_id];
        if let Some(m) = mask {
            inputs.push(m.value_id);
        }
        Ok(Self::trace_nary(
            OperationKind::ScaledDotProductAttention,
            inputs,
            &inner,
        ))
    }

    /// Delegates to `B::unfold`, recording an `OperationKind::Unfold` node.
    fn unfold<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::unfold(&t.inner, dim, size, step)?;
        Ok(Self::trace_unary(OperationKind::Unfold, t, &inner))
    }

    /// Delegates to `B::pixel_shuffle`, recording an `OperationKind::PixelShuffle` node.
    fn pixel_shuffle<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        upscale_factor: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::pixel_shuffle(&t.inner, upscale_factor)?;
        Ok(Self::trace_unary(OperationKind::PixelShuffle, t, &inner))
    }

    /// Delegates to `B::group_norm`, recording an `OperationKind::GroupNorm` node.
    fn group_norm<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        groups: usize,
        eps: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::group_norm(&t.inner, groups, eps)?;
        Ok(Self::trace_unary(OperationKind::GroupNorm, t, &inner))
    }

    /// Delegates to `B::instance_norm`, recording an `OperationKind::InstanceNorm` node.
    fn instance_norm<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        eps: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::instance_norm(&t.inner, eps)?;
        Ok(Self::trace_unary(OperationKind::InstanceNorm, t, &inner))
    }
}

impl<B: Backend + crate::tensor::backend::ModuleOps<B>> ModuleOps<Self> for TracingBackend<B> {
    /// Delegates to `B::conv1d`, additionally recording an
    /// `OperationKind::Conv1d` node with `strides`/`pads`/`dilations` attributes.
    fn conv1d<K: super::dtype::DType>(
        x: &<Self as StorageBackend>::Storage<K>,
        weight: &<Self as StorageBackend>::Storage<K>,
        bias: Option<&<Self as StorageBackend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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
            let out_id = g.add_value(shape_out.as_ref().to_vec(), DTypeId::F32, None);
            let mut inputs = vec![x.value_id, weight.value_id];
            if let Some(b) = bias {
                inputs.push(b.value_id);
            }
            let mut attrs = alloc::collections::BTreeMap::new();
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

            g.add_node(OperationKind::Conv1d, inputs, vec![out_id], attrs);
            out_id
        };
        Ok(TracingTensor { inner, value_id })
    }

    /// Delegates to `B::conv2d`, additionally recording an
    /// `OperationKind::Conv2d` node with `strides`/`pads`/`dilations` attributes.
    fn conv2d<K: super::dtype::DType>(
        x: &<Self as StorageBackend>::Storage<K>,
        weight: &<Self as StorageBackend>::Storage<K>,
        bias: Option<&<Self as StorageBackend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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
            let out_id = g.add_value(shape_out.as_ref().to_vec(), DTypeId::F32, None);
            let mut inputs = vec![x.value_id, weight.value_id];
            if let Some(b) = bias {
                inputs.push(b.value_id);
            }
            let mut attrs = alloc::collections::BTreeMap::new();
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

            g.add_node(OperationKind::Conv2d, inputs, vec![out_id], attrs);
            out_id
        };
        Ok(TracingTensor { inner, value_id })
    }

    /// Delegates to `B::max_pool2d`, additionally recording an `OperationKind::MaxPool2d` node.
    fn max_pool2d<K: super::dtype::DType>(
        x: &<Self as StorageBackend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::max_pool2d(&x.inner, kernel_size, stride, padding, dilation)?;
        Ok(Self::trace_unary(OperationKind::MaxPool2d, x, &inner))
    }

    /// Delegates to `B::avg_pool2d`, additionally recording an `OperationKind::AvgPool2d` node.
    fn avg_pool2d<K: super::dtype::DType>(
        x: &<Self as StorageBackend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::avg_pool2d(&x.inner, kernel_size, stride, padding)?;
        Ok(Self::trace_unary(OperationKind::AvgPool2d, x, &inner))
    }

    /// Delegates to `B::adaptive_avg_pool2d`, additionally recording an `OperationKind::AdaptiveAvgPool2d` node.
    fn adaptive_avg_pool2d<K: super::dtype::DType>(
        x: &<Self as StorageBackend>::Storage<K>,
        output_size: (usize, usize),
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::adaptive_avg_pool2d(&x.inner, output_size)?;
        Ok(Self::trace_unary(
            OperationKind::AdaptiveAvgPool2d,
            x,
            &inner,
        ))
    }

    /// Delegates to `B::embedding`, additionally recording an `OperationKind::Embedding` node.
    fn embedding<K: super::dtype::DType, KInt: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<KInt>,
        w: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::embedding(&t.inner, &w.inner)?;
        let shape = B::shape(&inner);
        let value_id = {
            let mut g = TRACING_GRAPH.lock();
            let out_id = g.add_value(shape.as_ref().to_vec(), DTypeId::F32, None);
            g.add_node(
                OperationKind::Embedding,
                vec![t.value_id, w.value_id],
                vec![out_id],
                alloc::collections::BTreeMap::new(),
            );
            out_id
        };
        Ok(TracingTensor { inner, value_id })
    }

    /// Delegates to `B::layer_norm`, additionally recording an `OperationKind::LayerNorm` node.
    fn layer_norm<K: super::dtype::DType>(
        x: &<Self as StorageBackend>::Storage<K>,
        weight: &<Self as StorageBackend>::Storage<K>,
        bias: Option<&<Self as StorageBackend>::Storage<K>>,
        eps: f32,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::layer_norm(&x.inner, &weight.inner, bias.map(|b| &b.inner), eps)?;
        Ok(Self::trace_unary(OperationKind::LayerNorm, x, &inner))
    }

    /// Delegates to `B::batch_norm`, additionally recording an `OperationKind::BatchNorm` node.
    fn batch_norm<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        w: Option<&<Self as StorageBackend>::Storage<K>>,
        b: Option<&<Self as StorageBackend>::Storage<K>>,
        rm: Option<&<Self as StorageBackend>::Storage<K>>,
        rv: Option<&<Self as StorageBackend>::Storage<K>>,
        e: f32,
        momentum: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::batch_norm(
            &t.inner,
            w.map(|x| &x.inner),
            b.map(|x| &x.inner),
            rm.map(|x| &x.inner),
            rv.map(|x| &x.inner),
            e,
            momentum,
        )?;
        Ok(Self::trace_unary(OperationKind::BatchNorm, t, &inner))
    }

    /// Delegates to `B::conv_transpose2d`, additionally recording an `OperationKind::ConvTranspose2d` node.
    fn conv_transpose2d<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        w: &<Self as StorageBackend>::Storage<K>,
        b: Option<&<Self as StorageBackend>::Storage<K>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
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
        Ok(Self::trace_unary(OperationKind::ConvTranspose2d, t, &inner))
    }
}

impl<B: Backend + crate::tensor::backend::LossOps<B>> LossOps<Self> for TracingBackend<B> {
    /// Delegates to `B::cross_entropy_loss`, additionally recording an `OperationKind::CrossEntropyLoss` node.
    fn cross_entropy_loss<K: super::dtype::DType, KInt: super::dtype::DType>(
        logits: &<Self as StorageBackend>::Storage<K>,
        targets: &<Self as StorageBackend>::Storage<KInt>,
        reduction: crate::nn::loss::Reduction,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::cross_entropy_loss(&logits.inner, &targets.inner, reduction)?;
        Ok(Self::trace_binary(
            OperationKind::CrossEntropyLoss,
            logits,
            targets,
            &inner,
        ))
    }

    /// Delegates to `B::mse_loss`, additionally recording an `OperationKind::MseLoss` node.
    fn mse_loss<K: super::dtype::DType>(
        predictions: &<Self as StorageBackend>::Storage<K>,
        targets: &<Self as StorageBackend>::Storage<K>,
        reduction: crate::nn::loss::Reduction,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::mse_loss(&predictions.inner, &targets.inner, reduction)?;
        Ok(Self::trace_binary(
            OperationKind::MseLoss,
            predictions,
            targets,
            &inner,
        ))
    }

    /// Delegates to `B::l1_loss`, additionally recording an `OperationKind::L1Loss` node.
    fn l1_loss<K: super::dtype::DType>(
        predictions: &<Self as StorageBackend>::Storage<K>,
        targets: &<Self as StorageBackend>::Storage<K>,
        reduction: crate::nn::loss::Reduction,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::l1_loss(&predictions.inner, &targets.inner, reduction)?;
        Ok(Self::trace_binary(
            OperationKind::L1Loss,
            predictions,
            targets,
            &inner,
        ))
    }

    /// Delegates to `B::bce_with_logits_loss`, additionally recording an `OperationKind::BceWithLogitsLoss` node.
    fn bce_with_logits_loss<K: super::dtype::DType>(
        logits: &<Self as StorageBackend>::Storage<K>,
        targets: &<Self as StorageBackend>::Storage<K>,
        _r: crate::nn::loss::Reduction,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::bce_with_logits_loss(&logits.inner, &targets.inner, _r)?;
        Ok(Self::trace_binary(
            OperationKind::BceWithLogitsLoss,
            logits,
            targets,
            &inner,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::backend::dummy::DummyBackend;
    use crate::tensor::device::Cpu;

    type B = TracingBackend<DummyBackend<Cpu>>;

    /// Looks up the `OperationKind` of the node that produced `value_id`, robust
    /// to other tests concurrently adding unrelated nodes to the shared
    /// process-wide `TRACING_GRAPH`.
    fn op_for_output(value_id: ValueId) -> OperationKind {
        let g = TRACING_GRAPH.lock();
        g.nodes
            .iter()
            .find(|n| n.outputs.contains(&value_id))
            .unwrap_or_else(|| panic!("no node found producing value {value_id}"))
            .op
    }

    #[test]
    /// Each unary activation must record its own `OperationKind`, not a
    /// different op's — regression test for a bug where abs/exp/neg/
    /// sqrt/log/tanh/sigmoid/swish all recorded `OperationKind::Relu` (copy-paste
    /// from `relu`'s implementation), silently corrupting ONNX export /
    /// graph visualization for any model using those activations.
    fn unary_activations_record_their_own_op_type() {
        let t: <B as StorageBackend>::Storage<f32> = TracingTensor {
            inner: alloc::vec![2, 3],
            value_id: 0,
        };

        let cases: [(
            fn(&<B as StorageBackend>::Storage<f32>) -> Result<<B as StorageBackend>::Storage<f32>>,
            OperationKind,
        ); 9] = [
            (<B as FloatOps<B>>::relu::<f32>, OperationKind::Relu),
            (<B as FloatOps<B>>::abs::<f32>, OperationKind::Abs),
            (<B as FloatOps<B>>::exp::<f32>, OperationKind::Exp),
            (<B as FloatOps<B>>::neg::<f32>, OperationKind::Neg),
            (<B as FloatOps<B>>::sqrt::<f32>, OperationKind::Sqrt),
            (<B as FloatOps<B>>::log::<f32>, OperationKind::Log),
            (<B as FloatOps<B>>::tanh::<f32>, OperationKind::Tanh),
            (<B as FloatOps<B>>::sigmoid::<f32>, OperationKind::Sigmoid),
            (<B as FloatOps<B>>::swish::<f32>, OperationKind::Swish),
        ];

        for (f, expected_op) in cases {
            let out = f(&t).unwrap();
            assert_eq!(
                op_for_output(out.value_id),
                expected_op,
                "wrong OperationKind recorded for this activation"
            );
        }
    }
}
