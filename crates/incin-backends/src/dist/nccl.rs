//! Two-rank, process-per-rank NCCL transport.
//!
//! The transport deliberately has a rank-local API: unlike the deterministic
//! reference backend, one process owns one input buffer and NCCL supplies the
//! other rank over the network. A fixed-size TCP bootstrap exchanges the NCCL
//! unique id and the [`PlanSummary`] before either process initializes its
//! communicator.

use core::ffi::c_char;
use core::marker::PhantomData;
use std::io::{self, Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::thread;
use std::time::{Duration, Instant};

use cudarc::driver::{CudaContext, CudaEvent, CudaSlice, CudaStream, DeviceRepr};
use cudarc::nccl::{Comm, Id, NcclType};
use incin_core::dist::mesh::{
    DeviceIdentity, LinkClass, MeshId, ProcessLayout, TopologyProbe, TransportVersion,
};
use incin_core::dist::placement::Placement;
use incin_core::dist::{
    AgreedPlan, CollectiveDType, CollectiveDescriptor, CollectiveError, CollectiveKind,
    CollectivePlan, ContextError, DataParallelDType, DataParallelError, DistributedContext,
    DistributedContextHandle, GradientId, GroupId, PipelineBoundaryId, PipelineDType,
    PipelineError, PipelineTransfer, PlanError, PlanSummary, RendezvousEndpoint, SequenceToken,
    StreamId, TensorParallelCollective, TensorParallelDType, TensorParallelError, TensorParallelId,
    preflight, validate_collective_dtype, validate_data_parallel_dtype, validate_pipeline_dtype,
    validate_tensor_parallel_dtype,
};
use incin_core::exec::ReduceOp;
use incin_core::shapes::{OperationKind, Shape};
use incin_core::tensor::base::Tensor;
use incin_core::tensor::device::{Device, DeviceId};
use incin_core::tensor::dtype::{DType, DTypeId};
use incin_core::tensor::grad::RequiresGrad;

use crate::cuda::backend::CudaBackendImpl;
use crate::cuda::storage::{CudaBuffer, CudaStorage};
use crate::cuda::tape::CudaGrads;

const WORLD: usize = 2;
const UNIQUE_ID_BYTES: usize = 128;
const MAGIC: [u8; 8] = *b"INCINN01";
const WIRE_BYTES: usize = 8 + 1 + 1 + 6 + 8 + 8 + 8 + UNIQUE_ID_BYTES;
const TOPOLOGY_MAGIC: [u8; 8] = *b"INCINT01";
const PERSISTENT_BYTES: usize = 64;
const ARCHITECTURE_BYTES: usize = 32;
const LIBRARY_BYTES: usize = 16;
const TOPOLOGY_WIRE_BYTES: usize = 8
    + 1
    + 1
    + 6
    + 4
    + 4
    + 4
    + 2
    + 2
    + 2
    + 2
    + PERSISTENT_BYTES
    + ARCHITECTURE_BYTES
    + LIBRARY_BYTES;

/// Which side of the two-rank TCP bootstrap this process owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootstrapRole {
    /// Rank zero listens at this address and creates the NCCL unique id.
    Root { bind: SocketAddr },
    /// Rank one connects to rank zero at this address.
    Peer { root: SocketAddr },
}

impl BootstrapRole {
    const fn rank(self) -> usize {
        match self {
            Self::Root { .. } => 0,
            Self::Peer { .. } => 1,
        }
    }
}

/// TCP bootstrap settings for exactly two network-accessible CUDA ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TwoRankBootstrapConfig {
    role: BootstrapRole,
    timeout: Duration,
}

impl TwoRankBootstrapConfig {
    /// Configure rank zero.
    #[must_use]
    pub const fn root(bind: SocketAddr, timeout: Duration) -> Self {
        Self {
            role: BootstrapRole::Root { bind },
            timeout,
        }
    }

    /// Configure rank one.
    #[must_use]
    pub const fn peer(root: SocketAddr, timeout: Duration) -> Self {
        Self {
            role: BootstrapRole::Peer { root },
            timeout,
        }
    }

    /// This process's rank.
    #[must_use]
    pub const fn rank(self) -> usize {
        self.role.rank()
    }

    /// Startup and socket I/O deadline.
    #[must_use]
    pub const fn timeout(self) -> Duration {
        self.timeout
    }

    /// Root/peer role and network address.
    #[must_use]
    pub const fn role(self) -> BootstrapRole {
        self.role
    }
}

/// Physical topology shared by both process-per-rank NCCL participants.
///
/// A launcher gathers one [`probe_local_cuda_identity`](Self::probe_local_cuda_identity)
/// result from each host, preserves rank order, and gives the same pair back to
/// both processes. Both ranks then derive one [`MeshId`] even when each host
/// exposes its local GPU as CUDA ordinal zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NcclTopology {
    identities: [DeviceIdentity; WORLD],
    rank: usize,
    transport: TransportVersion,
}

impl NcclTopology {
    /// Discover both hosts' physical identities over the bootstrap socket.
    ///
    /// This is the first of two bounded TCP sessions: discovery supplies the
    /// shared topology needed to bind a mesh and build a plan; later
    /// [`NcclTransport::connect`] exchanges that plan and the NCCL unique id.
    pub fn discover(
        config: TwoRankBootstrapConfig,
        device_ordinal: usize,
    ) -> Result<Self, NcclTransportError> {
        if config.timeout.is_zero() {
            return Err(NcclTransportError::InvalidTimeout);
        }
        let identity = Self::probe_local_cuda_identity(config.rank(), device_ordinal)?;
        let transport = Self::installed_transport_version()?;
        exchange_topology(config, identity, transport)
    }

    /// Discover topology from an agreed launcher context.
    ///
    /// A failure invalidates every clone and backend handle derived from the
    /// context, matching the same fail-stop rule as communicator creation.
    pub fn discover_context<M, R>(
        context: &DistributedContext<M, R>,
    ) -> Result<Self, NcclTransportError> {
        context.ensure_active()?;
        let config = bootstrap_from_context(context);
        let handle = context.handle();
        match Self::discover(config, context.local_cuda_device()) {
            Ok(topology) => Ok(topology),
            Err(error) => {
                handle.invalidate();
                Err(error)
            }
        }
    }

    /// Build the topology after the launcher has exchanged both identities.
    pub fn new(
        identities: [DeviceIdentity; WORLD],
        rank: usize,
        transport: TransportVersion,
    ) -> Result<Self, NcclTransportError> {
        if rank >= WORLD {
            return Err(NcclTransportError::LocalRank { rank, world: WORLD });
        }
        Ok(Self {
            identities,
            rank,
            transport,
        })
    }

    /// Query this process's stable CUDA UUID and compute capability.
    ///
    /// `rank` becomes the logical mesh ordinal. `device_ordinal` is local to
    /// this host after its CUDA visibility mask is applied.
    pub fn probe_local_cuda_identity(
        rank: usize,
        device_ordinal: usize,
    ) -> Result<DeviceIdentity, NcclTransportError> {
        if rank >= WORLD {
            return Err(NcclTransportError::LocalRank { rank, world: WORLD });
        }
        let context = CudaContext::new(device_ordinal)
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        let uuid = context
            .uuid()
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        let (major, minor) = context
            .compute_capability()
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        Ok(DeviceIdentity::new(
            DeviceId::cuda(rank),
            format_cuda_uuid(uuid.bytes),
            format!("sm_{major}{minor}"),
        ))
    }

    /// Query the dynamically loaded NCCL library version.
    pub fn installed_transport_version() -> Result<TransportVersion, NcclTransportError> {
        let encoded = catch_nccl_panic("query version", cudarc::nccl::result::get_nccl_version)?
            .map_err(nccl_error)?;
        let encoded =
            u32::try_from(encoded).map_err(|_| NcclTransportError::InvalidNcclVersion(encoded))?;
        Ok(TransportVersion::new(
            "nccl".to_string(),
            encoded / 10_000,
            (encoded / 100) % 100,
            encoded % 100,
        ))
    }
}

impl TopologyProbe for NcclTopology {
    fn identify(&self, device: DeviceId) -> Option<DeviceIdentity> {
        self.identities
            .iter()
            .find(|identity| identity.device() == device)
            .cloned()
    }

    fn link(&self, from: DeviceId, to: DeviceId) -> LinkClass {
        let known = |device| {
            self.identities
                .iter()
                .any(|identity| identity.device() == device)
        };
        if !known(from) || !known(to) {
            LinkClass::Unreachable
        } else if from == to {
            LinkClass::SameDevice
        } else {
            LinkClass::Network
        }
    }

    fn transport(&self) -> TransportVersion {
        self.transport.clone()
    }

    fn layout(&self) -> ProcessLayout {
        ProcessLayout::ProcessPerRank {
            rank: self.rank,
            world: WORLD,
        }
    }
}

/// Rank-local CUDA bytes indexed by a static dtype or [`Dyn`](incin_core::shapes::Dyn).
#[derive(Debug)]
pub struct NcclBuffer<K: DType> {
    data: CudaSlice<u8>,
    elements: usize,
    dtype: K::Field,
    marker: PhantomData<fn() -> K>,
}

impl<K: DType> NcclBuffer<K> {
    /// Join a device allocation to checked element and dtype metadata.
    pub fn try_from_device_bytes(
        data: CudaSlice<u8>,
        elements: usize,
        dtype: K::Field,
    ) -> Result<Self, NcclTransportError> {
        let runtime_dtype = K::descriptor(&dtype).builtin_id().ok_or_else(|| {
            NcclTransportError::InvalidBuffer("custom dtype not supported".to_string())
        })?;
        validate_collective_dtype(runtime_dtype)?;
        let expected = runtime_dtype
            .size_bytes(elements, OperationKind::Storage)
            .map_err(|error| NcclTransportError::InvalidBuffer(error.to_string()))?;
        if data.len() != expected {
            return Err(NcclTransportError::BufferBytes {
                expected,
                found: data.len(),
            });
        }
        Ok(Self {
            data,
            elements,
            dtype,
            marker: PhantomData,
        })
    }

    /// Runtime dtype after resolving `K`.
    #[must_use]
    pub fn dtype(&self) -> DTypeId {
        K::descriptor(&self.dtype)
            .builtin_id()
            .expect("built-in dtype")
    }

    /// Logical element count.
    #[must_use]
    pub const fn elements(&self) -> usize {
        self.elements
    }

    /// Physical byte count.
    #[must_use]
    pub fn bytes(&self) -> usize {
        self.data.len()
    }

    /// Borrow the underlying CUDA byte allocation.
    #[must_use]
    pub const fn device_bytes(&self) -> &CudaSlice<u8> {
        &self.data
    }

    /// Consume the wrapper and return its CUDA allocation.
    #[must_use]
    pub fn into_device_bytes(self) -> CudaSlice<u8> {
        self.data
    }
}

/// Completion event for one ordered NCCL launch.
#[derive(Debug)]
pub struct NcclEvent {
    event: CudaEvent,
    group: GroupId,
    sequence: SequenceToken,
    stream: StreamId,
    kind: CollectiveKind,
    distributed_context: Option<DistributedContextHandle>,
}

impl NcclEvent {
    /// Ordered communicator used by the launch.
    #[must_use]
    pub const fn group(&self) -> GroupId {
        self.group
    }

    /// Plan sequence executed by the launch.
    #[must_use]
    pub const fn sequence(&self) -> SequenceToken {
        self.sequence
    }

    /// Logical stream recorded by the plan.
    #[must_use]
    pub const fn stream(&self) -> StreamId {
        self.stream
    }

    /// Collective operation.
    #[must_use]
    pub const fn kind(&self) -> CollectiveKind {
        self.kind
    }

    /// Block until CUDA reports completion or failure.
    pub fn wait(&self) -> Result<(), NcclTransportError> {
        if let Some(handle) = &self.distributed_context {
            handle.ensure_active()?;
        }
        let result = self
            .event
            .synchronize()
            .map_err(|error| NcclTransportError::Cuda(error.to_string()));
        self.finish_wait(result)
    }

    /// Poll completion without allowing a missing/dead rank to block forever.
    pub fn wait_timeout(&self, timeout: Duration) -> Result<(), NcclTransportError> {
        if let Some(handle) = &self.distributed_context {
            handle.ensure_active()?;
        }
        let result = (|| {
            let deadline = Instant::now()
                .checked_add(timeout)
                .ok_or(NcclTransportError::InvalidTimeout)?;
            while !self.event.is_complete() {
                if Instant::now() >= deadline {
                    return Err(NcclTransportError::Timeout {
                        phase: "collective completion",
                        timeout,
                    });
                }
                thread::sleep(Duration::from_millis(1));
            }
            self.event
                .synchronize()
                .map_err(|error| NcclTransportError::Cuda(error.to_string()))
        })();
        self.finish_wait(result)
    }

    fn finish_wait(
        &self,
        result: Result<(), NcclTransportError>,
    ) -> Result<(), NcclTransportError> {
        if result.is_err()
            && let Some(handle) = &self.distributed_context
        {
            handle.invalidate();
        }
        result
    }
}

/// One NCCL communicator bound to one agreed plan and one of two ranks.
#[derive(Debug)]
pub struct NcclTransport {
    comm: Comm,
    context: std::sync::Arc<CudaContext>,
    stream: std::sync::Arc<CudaStream>,
    plan: CollectivePlan,
    agreed: AgreedPlan,
    cursor: usize,
    rank: usize,
    device_ordinal: usize,
    distributed_context: Option<DistributedContextHandle>,
}

impl NcclTransport {
    /// Bootstrap over TCP and initialize a two-rank NCCL communicator.
    ///
    /// Rank zero must use [`TwoRankBootstrapConfig::root`] and rank one must
    /// use [`TwoRankBootstrapConfig::peer`] with the root's reachable address.
    /// Plan agreement happens before `ncclCommInitRank`.
    pub fn connect(
        config: TwoRankBootstrapConfig,
        plan: CollectivePlan,
        device_ordinal: usize,
    ) -> Result<Self, NcclTransportError> {
        Self::connect_inner(config, plan, device_ordinal, None)
    }

    /// Initialize NCCL from an already-agreed distributed context.
    ///
    /// Rank, reachable root address, deadline, and process-local CUDA ordinal
    /// come from rendezvous rather than being repeated by the caller. Any
    /// communicator initialization or later launch failure invalidates the
    /// context's shared fail-stop handle.
    pub fn connect_context<M, R>(
        context: &DistributedContext<M, R>,
        plan: CollectivePlan,
    ) -> Result<Self, NcclTransportError> {
        context.ensure_active()?;
        let config = bootstrap_from_context(context);
        let handle = context.handle();
        match Self::connect_inner(
            config,
            plan,
            context.local_cuda_device(),
            Some(handle.clone()),
        ) {
            Ok(transport) => Ok(transport),
            Err(error) => {
                handle.invalidate();
                Err(error)
            }
        }
    }

    fn connect_inner(
        config: TwoRankBootstrapConfig,
        plan: CollectivePlan,
        device_ordinal: usize,
        distributed_context: Option<DistributedContextHandle>,
    ) -> Result<Self, NcclTransportError> {
        if config.timeout.is_zero() {
            return Err(NcclTransportError::InvalidTimeout);
        }
        let root_id = match config.role {
            BootstrapRole::Root { .. } => {
                Some(catch_nccl_panic("create unique id", Id::new)?.map_err(nccl_error)?)
            }
            BootstrapRole::Peer { .. } => None,
        };
        let root_bytes = root_id.as_ref().map(id_to_bytes);
        let bootstrap = exchange_bootstrap(config, plan.summary(), root_bytes)?;
        let id = root_id.unwrap_or_else(|| id_from_bytes(bootstrap.unique_id));
        let context = crate::cuda::gpu::cuda_cache::try_get_cuda_device(device_ordinal)
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        let stream = context.default_stream();
        let rank = config.rank();
        let comm = catch_nccl_panic("initialize communicator", || {
            Comm::from_rank(stream.clone(), rank, WORLD, id)
        })?
        .map_err(nccl_error)?;
        Ok(Self {
            comm,
            context,
            stream,
            plan,
            agreed: bootstrap.agreed,
            cursor: 0,
            rank,
            device_ordinal,
            distributed_context,
        })
    }

    /// This process's communicator rank.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Fixed communicator cardinality.
    #[must_use]
    pub const fn world_size(&self) -> usize {
        WORLD
    }

    /// Cross-rank plan agreement established by bootstrap.
    #[must_use]
    pub const fn agreed_plan(&self) -> AgreedPlan {
        self.agreed
    }

    /// Number of descriptors already submitted.
    #[must_use]
    pub const fn cursor(&self) -> usize {
        self.cursor
    }

    /// Launch the next descriptor for a statically checked dtype or for `Dyn`.
    ///
    /// Static unsupported dtypes cannot satisfy [`CollectiveDType`]. `Dyn`
    /// reaches the same method and is checked against the descriptor at runtime.
    pub fn execute<K: CollectiveDType>(
        &mut self,
        input: &NcclBuffer<K>,
    ) -> Result<(NcclBuffer<K>, NcclEvent), NcclTransportError> {
        let mut context_guard = ContextOperationGuard::new(&self.distributed_context)?;
        let descriptor =
            self.plan
                .descriptors()
                .get(self.cursor)
                .ok_or(NcclTransportError::PlanExhausted {
                    collectives: self.plan.descriptors().len(),
                })?;
        validate_launch(
            descriptor,
            self.cursor,
            input.dtype(),
            input.elements(),
            input.bytes(),
        )?;

        let mut output = self
            .stream
            .alloc_zeros::<u8>(descriptor.output_bytes())
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        launch_by_dtype(
            &self.comm,
            &self.stream,
            self.rank,
            descriptor.kind(),
            descriptor.dtype(),
            input.device_bytes(),
            &mut output,
            descriptor.input_elements(),
            descriptor.output_elements(),
        )?;
        let event = self
            .stream
            .record_event(None)
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        let result = NcclBuffer::try_from_device_bytes(
            output,
            descriptor.output_elements(),
            input.dtype.clone(),
        )?;
        let completion = NcclEvent {
            event,
            group: descriptor.group(),
            sequence: descriptor.sequence(),
            stream: descriptor.stream(),
            kind: descriptor.kind(),
            distributed_context: self.distributed_context.clone(),
        };
        self.cursor += 1;
        context_guard.commit();
        Ok((result, completion))
    }

    /// Mean-reduce one parameter gradient from a typed CUDA tensor.
    ///
    /// `K` is inferred from the tensor. Static integer and quantized tensors
    /// cannot satisfy [`DataParallelDType`]; a `Tensor<..., Dyn, ...>` reaches
    /// this method and its runtime dtype is checked before allocation or
    /// launch. The plan descriptor must carry the same stable [`GradientId`]
    /// and DP=2 `Partial<Mean> -> Replicated` semantics.
    pub fn synchronize_gradient<S, D, K, G, P>(
        &mut self,
        id: GradientId,
        parameter: &Tensor<S, CudaBackendImpl<D>, K, G, P>,
        gradients: &mut CudaGrads,
    ) -> Result<NcclEvent, NcclTransportError>
    where
        S: Shape,
        D: Device,
        K: DataParallelDType,
        G: RequiresGrad,
        P: Placement,
    {
        let mut context_guard = ContextOperationGuard::new(&self.distributed_context)?;
        let dtype = parameter
            .builtin_dtype_id()
            .ok_or(NcclTransportError::Collective(
                CollectiveError::UnsupportedDType {
                    dtype: parameter.dtype().builtin_id().unwrap_or(DTypeId::F32),
                },
            ))?;
        validate_data_parallel_dtype(dtype)?;
        let parameter_storage = parameter.inner();
        let input = gradients
            .get(parameter_storage.id)
            .cloned()
            .ok_or(NcclTransportError::MissingGradient { id })?;
        let (output, event) = self.execute_gradient_storage(id, dtype, &input)?;
        gradients.grads.insert(parameter_storage.id, output);
        context_guard.commit();
        Ok(event)
    }

    /// Execute one TP=2 linear/attention collective directly from CUDA tensor
    /// storage and return its flat rank-ordered result.
    ///
    /// Static integer and quantized tensors cannot satisfy
    /// [`TensorParallelDType`]. `Dyn` reaches the same method and is checked
    /// before allocation or launch. For all-gather, the flat result is ordered
    /// by rank shard; a higher layer must reassemble non-leading tensor axes
    /// before presenting the result as an ordinary row-major tensor.
    pub fn execute_tensor_parallel_flat<S, D, K, G, P>(
        &mut self,
        id: TensorParallelId,
        collective: TensorParallelCollective,
        input: &Tensor<S, CudaBackendImpl<D>, K, G, P>,
    ) -> Result<(CudaStorage, NcclEvent), NcclTransportError>
    where
        S: Shape,
        D: Device,
        K: TensorParallelDType,
        G: RequiresGrad,
        P: Placement,
    {
        let mut context_guard = ContextOperationGuard::new(&self.distributed_context)?;
        let dtype = input
            .builtin_dtype_id()
            .ok_or(NcclTransportError::Collective(
                CollectiveError::UnsupportedDType {
                    dtype: input.dtype().builtin_id().unwrap_or(DTypeId::F32),
                },
            ))?;
        validate_tensor_parallel_dtype(dtype)?;
        let input = input.inner();
        let descriptor =
            self.plan
                .descriptors()
                .get(self.cursor)
                .ok_or(NcclTransportError::PlanExhausted {
                    collectives: self.plan.descriptors().len(),
                })?;
        validate_tensor_parallel_launch(
            descriptor,
            self.cursor,
            id,
            collective,
            dtype,
            input.buffer.len,
            input.buffer.data.len(),
        )?;
        if input.buffer.device_id != self.device_ordinal {
            return Err(NcclTransportError::TensorParallelDevice {
                expected: self.device_ordinal,
                found: input.buffer.device_id,
            });
        }
        if input.meta.offset_elements() != 0
            || input.meta.layout() != incin_core::exec::LayoutClass::Contiguous
        {
            return Err(NcclTransportError::TensorParallelLayout {
                offset: input.meta.offset_elements(),
                layout: input.meta.layout(),
            });
        }

        let mut output = self
            .stream
            .alloc_zeros::<u8>(descriptor.output_bytes())
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        launch_by_dtype(
            &self.comm,
            &self.stream,
            self.rank,
            descriptor.kind(),
            descriptor.dtype(),
            &input.buffer.data,
            &mut output,
            descriptor.input_elements(),
            descriptor.output_elements(),
        )?;
        let event = self
            .stream
            .record_event(None)
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        let buffer = CudaBuffer {
            len: descriptor.output_elements(),
            dtype: dtype.descriptor(),
            data: std::sync::Arc::new(output),
            device: self.context.clone(),
            device_id: self.device_ordinal,
        };
        let storage = CudaStorage::try_new(
            std::sync::Arc::new(buffer),
            vec![descriptor.output_elements()],
        )
        .map_err(|error| NcclTransportError::InvalidBuffer(error.to_string()))?;
        let completion = NcclEvent {
            event,
            group: descriptor.group(),
            sequence: descriptor.sequence(),
            stream: descriptor.stream(),
            kind: descriptor.kind(),
            distributed_context: self.distributed_context.clone(),
        };
        self.cursor += 1;
        context_guard.commit();
        Ok((storage, completion))
    }

    /// Execute one TP=2 collective and reassemble the replicated logical shape
    /// on CUDA.
    ///
    /// NCCL all-gather returns rank-major shards. When the sharded tensor axis
    /// is not leading, this method materializes the required axis movement on
    /// the same CUDA stream before recording the returned completion event.
    pub fn execute_tensor_parallel<S, D, K, G, P>(
        &mut self,
        id: TensorParallelId,
        collective: TensorParallelCollective,
        input: &Tensor<S, CudaBackendImpl<D>, K, G, P>,
        global_shape: &[usize],
    ) -> Result<(CudaStorage, NcclEvent), NcclTransportError>
    where
        S: Shape,
        D: Device,
        K: TensorParallelDType,
        G: RequiresGrad,
        P: Placement,
    {
        let mut context_guard = ContextOperationGuard::new(&self.distributed_context)?;
        let descriptor =
            self.plan
                .descriptors()
                .get(self.cursor)
                .ok_or(NcclTransportError::PlanExhausted {
                    collectives: self.plan.descriptors().len(),
                })?;
        let input_shape = input.inner().meta.shape().dims();
        let local_shape =
            validate_tensor_parallel_shapes(descriptor, collective, input_shape, global_shape)?;

        let (flat, first_event) = self.execute_tensor_parallel_flat(id, collective, input)?;
        let storage = reassemble_tensor_parallel_storage::<D, K>(
            &flat,
            collective,
            &local_shape,
            global_shape,
        )?;
        let event = self
            .stream
            .record_event(None)
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        context_guard.commit();
        Ok((
            storage,
            NcclEvent {
                event,
                group: first_event.group,
                sequence: first_event.sequence,
                stream: first_event.stream,
                kind: first_event.kind,
                distributed_context: first_event.distributed_context,
            },
        ))
    }

    /// Execute one PP=2 activation or gradient transfer from a typed CUDA
    /// tensor.
    ///
    /// Both ranks submit a same-shaped tensor. The source tensor is sent and
    /// copied to the returned source storage; the destination's input is a
    /// placeholder overwritten by `recv`. This symmetric call shape lets `K`
    /// be inferred on both processes while the immutable descriptor carries
    /// one global source/destination contract.
    pub fn execute_pipeline<S, D, K, G, P>(
        &mut self,
        boundary: PipelineBoundaryId,
        transfer: PipelineTransfer,
        microbatch: usize,
        input: &Tensor<S, CudaBackendImpl<D>, K, G, P>,
    ) -> Result<(CudaStorage, NcclEvent), NcclTransportError>
    where
        S: Shape,
        D: Device,
        K: PipelineDType,
        G: RequiresGrad,
        P: Placement,
    {
        let mut context_guard = ContextOperationGuard::new(&self.distributed_context)?;
        let dtype = input
            .builtin_dtype_id()
            .ok_or(NcclTransportError::Collective(
                CollectiveError::UnsupportedDType {
                    dtype: input.dtype().builtin_id().unwrap_or(DTypeId::F32),
                },
            ))?;
        validate_pipeline_dtype(dtype)?;
        let input = input.inner();
        let descriptor =
            self.plan
                .descriptors()
                .get(self.cursor)
                .ok_or(NcclTransportError::PlanExhausted {
                    collectives: self.plan.descriptors().len(),
                })?;
        validate_pipeline_launch(
            descriptor,
            self.cursor,
            boundary,
            transfer,
            microbatch,
            dtype,
            input.buffer.len,
            input.buffer.data.len(),
        )?;
        if input.buffer.device_id != self.device_ordinal {
            return Err(NcclTransportError::PipelineDevice {
                expected: self.device_ordinal,
                found: input.buffer.device_id,
            });
        }
        if input.meta.offset_elements() != 0
            || input.meta.layout() != incin_core::exec::LayoutClass::Contiguous
        {
            return Err(NcclTransportError::PipelineLayout {
                offset: input.meta.offset_elements(),
                layout: input.meta.layout(),
            });
        }

        let mut output = self
            .stream
            .alloc_zeros::<u8>(descriptor.output_bytes())
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        launch_by_dtype(
            &self.comm,
            &self.stream,
            self.rank,
            descriptor.kind(),
            descriptor.dtype(),
            &input.buffer.data,
            &mut output,
            descriptor.input_elements(),
            descriptor.output_elements(),
        )?;
        let event = self
            .stream
            .record_event(None)
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        let buffer = CudaBuffer {
            len: descriptor.output_elements(),
            dtype: dtype.descriptor(),
            data: std::sync::Arc::new(output),
            device: self.context.clone(),
            device_id: self.device_ordinal,
        };
        let storage = CudaStorage::try_new(
            std::sync::Arc::new(buffer),
            input.meta.shape().dims().to_vec(),
        )
        .map_err(|error| NcclTransportError::InvalidBuffer(error.to_string()))?;
        let completion = NcclEvent {
            event,
            group: descriptor.group(),
            sequence: descriptor.sequence(),
            stream: descriptor.stream(),
            kind: descriptor.kind(),
            distributed_context: self.distributed_context.clone(),
        };
        self.cursor += 1;
        context_guard.commit();
        Ok((storage, completion))
    }

    fn execute_gradient_storage(
        &mut self,
        id: GradientId,
        dtype: DTypeId,
        input: &CudaStorage,
    ) -> Result<(CudaStorage, NcclEvent), NcclTransportError> {
        let mut context_guard = ContextOperationGuard::new(&self.distributed_context)?;
        let descriptor =
            self.plan
                .descriptors()
                .get(self.cursor)
                .ok_or(NcclTransportError::PlanExhausted {
                    collectives: self.plan.descriptors().len(),
                })?;
        validate_gradient_launch(
            descriptor,
            self.cursor,
            id,
            dtype,
            input.buffer.len,
            input.buffer.data.len(),
        )?;
        if input.buffer.device_id != self.device_ordinal {
            return Err(NcclTransportError::GradientDevice {
                expected: self.device_ordinal,
                found: input.buffer.device_id,
            });
        }
        if input.meta.offset_elements() != 0
            || input.meta.layout() != incin_core::exec::LayoutClass::Contiguous
        {
            return Err(NcclTransportError::GradientLayout {
                offset: input.meta.offset_elements(),
                layout: input.meta.layout(),
            });
        }

        let mut output = self
            .stream
            .alloc_zeros::<u8>(descriptor.output_bytes())
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        launch_by_dtype(
            &self.comm,
            &self.stream,
            self.rank,
            descriptor.kind(),
            descriptor.dtype(),
            &input.buffer.data,
            &mut output,
            descriptor.input_elements(),
            descriptor.output_elements(),
        )?;
        let event = self
            .stream
            .record_event(None)
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        let buffer = CudaBuffer {
            len: descriptor.output_elements(),
            dtype: dtype.descriptor(),
            data: std::sync::Arc::new(output),
            device: self.context.clone(),
            device_id: self.device_ordinal,
        };
        let storage = CudaStorage::try_new(
            std::sync::Arc::new(buffer),
            input.meta.shape().dims().to_vec(),
        )
        .map_err(|error| NcclTransportError::InvalidBuffer(error.to_string()))?;
        let completion = NcclEvent {
            event,
            group: descriptor.group(),
            sequence: descriptor.sequence(),
            stream: descriptor.stream(),
            kind: descriptor.kind(),
            distributed_context: self.distributed_context.clone(),
        };
        self.cursor += 1;
        context_guard.commit();
        Ok((storage, completion))
    }
}

fn bootstrap_from_context<M, R>(context: &DistributedContext<M, R>) -> TwoRankBootstrapConfig {
    match context.endpoint() {
        RendezvousEndpoint::Root { bind } => TwoRankBootstrapConfig::root(bind, context.timeout()),
        RendezvousEndpoint::Peer { root } => TwoRankBootstrapConfig::peer(root, context.timeout()),
    }
}

struct ContextOperationGuard {
    handle: Option<DistributedContextHandle>,
    committed: bool,
}

impl ContextOperationGuard {
    fn new(handle: &Option<DistributedContextHandle>) -> Result<Self, NcclTransportError> {
        if let Some(handle) = handle {
            handle.ensure_active()?;
        }
        Ok(Self {
            handle: handle.clone(),
            committed: false,
        })
    }

    fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for ContextOperationGuard {
    fn drop(&mut self) {
        if !self.committed
            && let Some(handle) = &self.handle
        {
            handle.invalidate();
        }
    }
}

fn validate_gradient_launch(
    descriptor: &CollectiveDescriptor,
    cursor: usize,
    id: GradientId,
    dtype: DTypeId,
    elements: usize,
    bytes: usize,
) -> Result<(), NcclTransportError> {
    validate_data_parallel_dtype(dtype)?;
    if descriptor.tag().get() != id.get() {
        return Err(NcclTransportError::GradientIdentity {
            expected: descriptor.tag().get(),
            found: id.get(),
        });
    }
    let expected_source = incin_core::dist::PlacementKind::Partial {
        reduction: ReduceOp::Mean,
    };
    if descriptor.kind() != CollectiveKind::AllReduce(ReduceOp::Mean)
        || descriptor.source() != expected_source
        || descriptor.destination() != incin_core::dist::PlacementKind::Replicated
    {
        return Err(NcclTransportError::NotDataParallelGradient {
            kind: descriptor.kind(),
            from_placement: descriptor.source(),
            to_placement: descriptor.destination(),
        });
    }
    validate_launch(descriptor, cursor, dtype, elements, bytes)
}

#[allow(clippy::too_many_arguments)]
fn validate_tensor_parallel_launch(
    descriptor: &CollectiveDescriptor,
    cursor: usize,
    id: TensorParallelId,
    collective: TensorParallelCollective,
    dtype: DTypeId,
    elements: usize,
    bytes: usize,
) -> Result<(), NcclTransportError> {
    validate_tensor_parallel_dtype(dtype)?;
    let expected_tag = collective.plan_tag(id).get();
    if descriptor.tag().get() != expected_tag {
        return Err(NcclTransportError::TensorParallelIdentity {
            expected: descriptor.tag().get(),
            found: expected_tag,
        });
    }
    let valid = match collective {
        TensorParallelCollective::ColumnOutputGather { tensor_axis }
        | TensorParallelCollective::AttentionHeadGather { tensor_axis } => {
            descriptor.kind() == CollectiveKind::AllGather
                && descriptor.source()
                    == incin_core::dist::PlacementKind::Sharded { axis: tensor_axis }
                && descriptor.destination() == incin_core::dist::PlacementKind::Replicated
        }
        TensorParallelCollective::RowOutputSum => {
            descriptor.kind() == CollectiveKind::AllReduce(ReduceOp::Sum)
                && descriptor.source()
                    == incin_core::dist::PlacementKind::Partial {
                        reduction: ReduceOp::Sum,
                    }
                && descriptor.destination() == incin_core::dist::PlacementKind::Replicated
        }
    };
    if !valid {
        return Err(NcclTransportError::NotTensorParallelOperation {
            expected: collective,
            kind: descriptor.kind(),
            from_placement: descriptor.source(),
            to_placement: descriptor.destination(),
        });
    }
    validate_launch(descriptor, cursor, dtype, elements, bytes)
}

#[allow(clippy::too_many_arguments)]
fn validate_pipeline_launch(
    descriptor: &CollectiveDescriptor,
    cursor: usize,
    boundary: PipelineBoundaryId,
    transfer: PipelineTransfer,
    microbatch: usize,
    dtype: DTypeId,
    elements: usize,
    bytes: usize,
) -> Result<(), NcclTransportError> {
    validate_pipeline_dtype(dtype)?;
    let expected = transfer.plan_tag(boundary, microbatch).get();
    if descriptor.tag().get() != expected {
        return Err(NcclTransportError::PipelineIdentity {
            expected,
            found: descriptor.tag().get(),
        });
    }
    let expected_kind = CollectiveKind::SendRecv {
        source: transfer.source_rank(),
        destination: transfer.destination_rank(),
    };
    let expected_source = incin_core::dist::PlacementKind::PipelineStage {
        index: transfer.source_rank(),
    };
    let expected_destination = incin_core::dist::PlacementKind::PipelineStage {
        index: transfer.destination_rank(),
    };
    if descriptor.kind() != expected_kind
        || descriptor.source() != expected_source
        || descriptor.destination() != expected_destination
    {
        return Err(NcclTransportError::NotPipelineTransfer {
            expected: transfer,
            kind: descriptor.kind(),
            from_placement: descriptor.source(),
            to_placement: descriptor.destination(),
        });
    }
    validate_launch(descriptor, cursor, dtype, elements, bytes)
}

fn validate_tensor_parallel_shapes(
    descriptor: &CollectiveDescriptor,
    collective: TensorParallelCollective,
    input_shape: &[usize],
    global_shape: &[usize],
) -> Result<Vec<usize>, NcclTransportError> {
    let global_elements = checked_shape_elements(global_shape)?;
    if global_elements != descriptor.output_elements() {
        return Err(NcclTransportError::TensorParallelOutputElements {
            expected: descriptor.output_elements(),
            found: global_elements,
        });
    }

    let expected_local = match collective {
        TensorParallelCollective::ColumnOutputGather { tensor_axis }
        | TensorParallelCollective::AttentionHeadGather { tensor_axis } => {
            let Some(&global_extent) = global_shape.get(tensor_axis) else {
                return Err(TensorParallelError::AxisOutOfBounds {
                    axis: tensor_axis,
                    rank: global_shape.len(),
                }
                .into());
            };
            let local_extent = incin_core::dist::validate_two_way_extent(
                match collective {
                    TensorParallelCollective::ColumnOutputGather { .. } => {
                        incin_core::dist::TensorParallelDimension::OutputFeatures
                    }
                    TensorParallelCollective::AttentionHeadGather { .. } => {
                        incin_core::dist::TensorParallelDimension::AttentionHeads
                    }
                    TensorParallelCollective::RowOutputSum => unreachable!(),
                },
                global_extent,
            )?;
            let mut expected = global_shape.to_vec();
            expected[tensor_axis] = local_extent;
            expected
        }
        TensorParallelCollective::RowOutputSum => global_shape.to_vec(),
    };
    if input_shape != expected_local {
        return Err(NcclTransportError::TensorParallelShape {
            expected: expected_local,
            found: input_shape.to_vec(),
        });
    }
    Ok(expected_local)
}

fn reassemble_tensor_parallel_storage<D: Device, K: DType>(
    flat: &CudaStorage,
    collective: TensorParallelCollective,
    local_shape: &[usize],
    global_shape: &[usize],
) -> Result<CudaStorage, NcclTransportError>
where
    K: TensorParallelDType,
{
    let mut storage = match collective {
        TensorParallelCollective::ColumnOutputGather { tensor_axis }
        | TensorParallelCollective::AttentionHeadGather { tensor_axis } => {
            let mut rank_major = Vec::with_capacity(local_shape.len() + 1);
            rank_major.push(WORLD);
            rank_major.extend_from_slice(local_shape);
            let mut storage = CudaBackendImpl::<D>::reshape::<K>(flat, &rank_major)
                .map_err(|error| NcclTransportError::InvalidBuffer(error.to_string()))?;
            for position in 0..tensor_axis {
                storage = CudaBackendImpl::<D>::transpose::<K>(&storage, position, position + 1)
                    .map_err(|error| NcclTransportError::InvalidBuffer(error.to_string()))?;
            }
            storage
        }
        TensorParallelCollective::RowOutputSum => flat.clone(),
    };
    storage = CudaBackendImpl::<D>::reshape::<K>(&storage, global_shape)
        .map_err(|error| NcclTransportError::InvalidBuffer(error.to_string()))?;
    Ok(storage)
}

fn checked_shape_elements(shape: &[usize]) -> Result<usize, NcclTransportError> {
    shape.iter().copied().try_fold(1usize, |elements, extent| {
        elements
            .checked_mul(extent)
            .ok_or(NcclTransportError::TensorParallel(
                TensorParallelError::ElementCountOverflow,
            ))
    })
}

fn validate_launch(
    descriptor: &CollectiveDescriptor,
    cursor: usize,
    dtype: DTypeId,
    elements: usize,
    bytes: usize,
) -> Result<(), NcclTransportError> {
    if descriptor.sequence().get() != cursor as u64 {
        return Err(NcclTransportError::Sequence {
            expected: cursor as u64,
            found: descriptor.sequence().get(),
        });
    }
    if descriptor.group().ranks() != WORLD {
        return Err(NcclTransportError::GroupCardinality {
            expected: WORLD,
            found: descriptor.group().ranks(),
        });
    }
    validate_collective_dtype(dtype)?;
    if dtype != descriptor.dtype() {
        return Err(NcclTransportError::DType {
            expected: descriptor.dtype(),
            found: dtype,
        });
    }
    if elements != descriptor.input_elements() {
        return Err(NcclTransportError::Elements {
            expected: descriptor.input_elements(),
            found: elements,
        });
    }
    if bytes != descriptor.input_bytes() {
        return Err(NcclTransportError::BufferBytes {
            expected: descriptor.input_bytes(),
            found: bytes,
        });
    }
    validate_reduction(descriptor.kind(), descriptor.dtype())
}

fn exchange_topology(
    config: TwoRankBootstrapConfig,
    local_identity: DeviceIdentity,
    local_transport: TransportVersion,
) -> Result<NcclTopology, NcclTransportError> {
    let local = TopologyWire::new(config.rank() as u8, &local_identity, &local_transport)?;
    let remote = match config.role {
        BootstrapRole::Root { bind } => {
            let listener = TcpListener::bind(bind).map_err(|error| io_error("bind", error))?;
            listener
                .set_nonblocking(true)
                .map_err(|error| io_error("set nonblocking", error))?;
            let mut stream = accept_until(&listener, config.timeout)?;
            // This fixed-cardinality session accepts exactly one peer. Close
            // the listening socket before replying so rank one cannot queue
            // the next protocol phase against this old listener and receive a
            // reset when the function returns.
            drop(listener);
            configure_stream(&stream, config.timeout)?;
            let remote = read_topology_wire(&mut stream)?;
            validate_topology_wire(&remote, 1)?;
            write_topology_wire(&mut stream, local)?;
            remote
        }
        BootstrapRole::Peer { root } => {
            let mut stream = connect_until(root, config.timeout)?;
            configure_stream(&stream, config.timeout)?;
            write_topology_wire(&mut stream, local)?;
            let remote = read_topology_wire(&mut stream)?;
            validate_topology_wire(&remote, 0)?;
            remote
        }
    };
    let remote_transport = remote.transport()?;
    if remote_transport != local_transport {
        return Err(NcclTransportError::TransportMismatch {
            local: format_transport(&local_transport),
            remote: format_transport(&remote_transport),
        });
    }
    let remote_identity = remote.identity()?;
    let identities = if config.rank() == 0 {
        [local_identity, remote_identity]
    } else {
        [remote_identity, local_identity]
    };
    NcclTopology::new(identities, config.rank(), local_transport)
}

#[derive(Debug, Clone, Copy)]
struct TopologyWire {
    rank: u8,
    world: u8,
    major: u32,
    minor: u32,
    patch: u32,
    persistent_len: u16,
    architecture_len: u16,
    library_len: u16,
    persistent: [u8; PERSISTENT_BYTES],
    architecture: [u8; ARCHITECTURE_BYTES],
    library: [u8; LIBRARY_BYTES],
}

impl TopologyWire {
    fn new(
        rank: u8,
        identity: &DeviceIdentity,
        transport: &TransportVersion,
    ) -> Result<Self, NcclTransportError> {
        let (persistent, persistent_len) =
            fixed_string::<PERSISTENT_BYTES>("persistent CUDA identity", identity.persistent())?;
        let (architecture, architecture_len) =
            fixed_string::<ARCHITECTURE_BYTES>("CUDA architecture", identity.architecture())?;
        let (library, library_len) =
            fixed_string::<LIBRARY_BYTES>("transport library", transport.library())?;
        let (major, minor, patch) = transport.version();
        Ok(Self {
            rank,
            world: WORLD as u8,
            major,
            minor,
            patch,
            persistent_len,
            architecture_len,
            library_len,
            persistent,
            architecture,
            library,
        })
    }

    fn identity(self) -> Result<DeviceIdentity, NcclTransportError> {
        Ok(DeviceIdentity::new(
            DeviceId::cuda(self.rank as usize),
            decode_fixed(
                &self.persistent,
                self.persistent_len,
                "persistent CUDA identity",
            )?,
            decode_fixed(
                &self.architecture,
                self.architecture_len,
                "CUDA architecture",
            )?,
        ))
    }

    fn transport(self) -> Result<TransportVersion, NcclTransportError> {
        Ok(TransportVersion::new(
            decode_fixed(&self.library, self.library_len, "transport library")?,
            self.major,
            self.minor,
            self.patch,
        ))
    }

    fn encode(self) -> [u8; TOPOLOGY_WIRE_BYTES] {
        let mut bytes = [0; TOPOLOGY_WIRE_BYTES];
        bytes[..8].copy_from_slice(&TOPOLOGY_MAGIC);
        bytes[8] = self.rank;
        bytes[9] = self.world;
        bytes[16..20].copy_from_slice(&self.major.to_be_bytes());
        bytes[20..24].copy_from_slice(&self.minor.to_be_bytes());
        bytes[24..28].copy_from_slice(&self.patch.to_be_bytes());
        bytes[28..30].copy_from_slice(&self.persistent_len.to_be_bytes());
        bytes[30..32].copy_from_slice(&self.architecture_len.to_be_bytes());
        bytes[32..34].copy_from_slice(&self.library_len.to_be_bytes());
        let persistent_end = 36 + PERSISTENT_BYTES;
        let architecture_end = persistent_end + ARCHITECTURE_BYTES;
        bytes[36..persistent_end].copy_from_slice(&self.persistent);
        bytes[persistent_end..architecture_end].copy_from_slice(&self.architecture);
        bytes[architecture_end..].copy_from_slice(&self.library);
        bytes
    }

    fn decode(bytes: [u8; TOPOLOGY_WIRE_BYTES]) -> Result<Self, NcclTransportError> {
        if bytes[..8] != TOPOLOGY_MAGIC {
            return Err(NcclTransportError::Protocol(
                "topology bootstrap magic mismatch",
            ));
        }
        let read_u32 = |start: usize| {
            let mut value = [0; 4];
            value.copy_from_slice(&bytes[start..start + 4]);
            u32::from_be_bytes(value)
        };
        let read_u16 = |start: usize| {
            let mut value = [0; 2];
            value.copy_from_slice(&bytes[start..start + 2]);
            u16::from_be_bytes(value)
        };
        let persistent_end = 36 + PERSISTENT_BYTES;
        let architecture_end = persistent_end + ARCHITECTURE_BYTES;
        let mut persistent = [0; PERSISTENT_BYTES];
        persistent.copy_from_slice(&bytes[36..persistent_end]);
        let mut architecture = [0; ARCHITECTURE_BYTES];
        architecture.copy_from_slice(&bytes[persistent_end..architecture_end]);
        let mut library = [0; LIBRARY_BYTES];
        library.copy_from_slice(&bytes[architecture_end..]);
        Ok(Self {
            rank: bytes[8],
            world: bytes[9],
            major: read_u32(16),
            minor: read_u32(20),
            patch: read_u32(24),
            persistent_len: read_u16(28),
            architecture_len: read_u16(30),
            library_len: read_u16(32),
            persistent,
            architecture,
            library,
        })
    }
}

fn fixed_string<const N: usize>(
    field: &'static str,
    value: &str,
) -> Result<([u8; N], u16), NcclTransportError> {
    if value.len() > N || value.len() > u16::MAX as usize {
        return Err(NcclTransportError::FieldTooLong {
            field,
            maximum: N,
            found: value.len(),
        });
    }
    let mut bytes = [0; N];
    bytes[..value.len()].copy_from_slice(value.as_bytes());
    Ok((bytes, value.len() as u16))
}

fn decode_fixed<const N: usize>(
    bytes: &[u8; N],
    len: u16,
    field: &'static str,
) -> Result<String, NcclTransportError> {
    let len = usize::from(len);
    if len > N {
        return Err(NcclTransportError::FieldTooLong {
            field,
            maximum: N,
            found: len,
        });
    }
    std::str::from_utf8(&bytes[..len])
        .map(str::to_owned)
        .map_err(|_| NcclTransportError::Protocol("topology field is not UTF-8"))
}

fn validate_topology_wire(
    message: &TopologyWire,
    expected_rank: u8,
) -> Result<(), NcclTransportError> {
    if message.world as usize != WORLD {
        return Err(NcclTransportError::WorldSize {
            expected: WORLD,
            found: message.world as usize,
        });
    }
    if message.rank != expected_rank {
        return Err(NcclTransportError::RemoteRank {
            expected: expected_rank as usize,
            found: message.rank as usize,
        });
    }
    Ok(())
}

fn write_topology_wire(
    stream: &mut TcpStream,
    message: TopologyWire,
) -> Result<(), NcclTransportError> {
    stream
        .write_all(&message.encode())
        .map_err(|error| io_error("write topology bootstrap", error))
}

fn read_topology_wire(stream: &mut TcpStream) -> Result<TopologyWire, NcclTransportError> {
    let mut bytes = [0; TOPOLOGY_WIRE_BYTES];
    stream
        .read_exact(&mut bytes)
        .map_err(|error| io_error("read topology bootstrap", error))?;
    TopologyWire::decode(bytes)
}

fn format_transport(transport: &TransportVersion) -> String {
    let (major, minor, patch) = transport.version();
    format!("{} {major}.{minor}.{patch}", transport.library())
}

#[derive(Debug)]
struct BootstrapResult {
    unique_id: [u8; UNIQUE_ID_BYTES],
    agreed: AgreedPlan,
}

fn exchange_bootstrap(
    config: TwoRankBootstrapConfig,
    local: PlanSummary,
    root_id: Option<[u8; UNIQUE_ID_BYTES]>,
) -> Result<BootstrapResult, NcclTransportError> {
    if config.timeout.is_zero() {
        return Err(NcclTransportError::InvalidTimeout);
    }
    match config.role {
        BootstrapRole::Root { bind } => {
            let id = root_id.ok_or(NcclTransportError::MissingRootId)?;
            let listener = TcpListener::bind(bind).map_err(|error| io_error("bind", error))?;
            listener
                .set_nonblocking(true)
                .map_err(|error| io_error("set nonblocking", error))?;
            let mut stream = accept_until(&listener, config.timeout)?;
            drop(listener);
            configure_stream(&stream, config.timeout)?;
            let peer = read_wire(&mut stream)?;
            validate_wire(&peer, 1)?;
            let remote = peer.summary()?;
            let agreed = preflight(WORLD, &[local, remote])?;
            write_wire(&mut stream, WireMessage::new(0, local, id))?;
            Ok(BootstrapResult {
                unique_id: id,
                agreed,
            })
        }
        BootstrapRole::Peer { root } => {
            if root_id.is_some() {
                return Err(NcclTransportError::UnexpectedPeerId);
            }
            let mut stream = connect_until(root, config.timeout)?;
            configure_stream(&stream, config.timeout)?;
            write_wire(
                &mut stream,
                WireMessage::new(1, local, [0; UNIQUE_ID_BYTES]),
            )?;
            let root = read_wire(&mut stream)?;
            validate_wire(&root, 0)?;
            let remote = root.summary()?;
            let agreed = preflight(WORLD, &[remote, local])?;
            Ok(BootstrapResult {
                unique_id: root.unique_id,
                agreed,
            })
        }
    }
}

fn accept_until(
    listener: &TcpListener,
    timeout: Duration,
) -> Result<TcpStream, NcclTransportError> {
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or(NcclTransportError::InvalidTimeout)?;
    loop {
        match listener.accept() {
            Ok((stream, _)) => return Ok(stream),
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                if Instant::now() >= deadline {
                    return Err(NcclTransportError::Timeout {
                        phase: "accept rank one",
                        timeout,
                    });
                }
                thread::sleep(Duration::from_millis(2));
            }
            Err(error) => return Err(io_error("accept", error)),
        }
    }
}

fn connect_until(root: SocketAddr, timeout: Duration) -> Result<TcpStream, NcclTransportError> {
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or(NcclTransportError::InvalidTimeout)?;
    loop {
        let now = Instant::now();
        if now >= deadline {
            return Err(NcclTransportError::Timeout {
                phase: "connect to rank zero",
                timeout,
            });
        }
        let remaining = deadline.saturating_duration_since(now);
        match TcpStream::connect_timeout(&root, remaining.min(Duration::from_millis(100))) {
            Ok(stream) => return Ok(stream),
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::ConnectionRefused
                        | io::ErrorKind::TimedOut
                        | io::ErrorKind::WouldBlock
                ) =>
            {
                thread::sleep(Duration::from_millis(2));
            }
            Err(error) => return Err(io_error("connect", error)),
        }
    }
}

fn configure_stream(stream: &TcpStream, timeout: Duration) -> Result<(), NcclTransportError> {
    stream
        .set_read_timeout(Some(timeout))
        .map_err(|error| io_error("set read timeout", error))?;
    stream
        .set_write_timeout(Some(timeout))
        .map_err(|error| io_error("set write timeout", error))?;
    stream
        .set_nodelay(true)
        .map_err(|error| io_error("set TCP_NODELAY", error))
}

#[derive(Debug, Clone, Copy)]
struct WireMessage {
    rank: u8,
    world: u8,
    mesh: u64,
    hash: u64,
    collectives: u64,
    unique_id: [u8; UNIQUE_ID_BYTES],
}

impl WireMessage {
    fn new(rank: u8, summary: PlanSummary, unique_id: [u8; UNIQUE_ID_BYTES]) -> Self {
        Self {
            rank,
            world: WORLD as u8,
            mesh: summary.mesh_id().digest(),
            hash: summary.hash(),
            collectives: summary.collective_count() as u64,
            unique_id,
        }
    }

    fn summary(self) -> Result<PlanSummary, NcclTransportError> {
        let collectives = usize::try_from(self.collectives)
            .map_err(|_| NcclTransportError::Protocol("collective count exceeds usize"))?;
        Ok(PlanSummary::from_parts(
            MeshId::from_digest(self.mesh),
            self.hash,
            collectives,
        ))
    }

    fn encode(self) -> [u8; WIRE_BYTES] {
        let mut bytes = [0; WIRE_BYTES];
        bytes[..8].copy_from_slice(&MAGIC);
        bytes[8] = self.rank;
        bytes[9] = self.world;
        bytes[16..24].copy_from_slice(&self.mesh.to_be_bytes());
        bytes[24..32].copy_from_slice(&self.hash.to_be_bytes());
        bytes[32..40].copy_from_slice(&self.collectives.to_be_bytes());
        bytes[40..].copy_from_slice(&self.unique_id);
        bytes
    }

    fn decode(bytes: [u8; WIRE_BYTES]) -> Result<Self, NcclTransportError> {
        if bytes[..8] != MAGIC {
            return Err(NcclTransportError::Protocol("bootstrap magic mismatch"));
        }
        let read_u64 = |start: usize| {
            let mut value = [0; 8];
            value.copy_from_slice(&bytes[start..start + 8]);
            u64::from_be_bytes(value)
        };
        let mut unique_id = [0; UNIQUE_ID_BYTES];
        unique_id.copy_from_slice(&bytes[40..]);
        Ok(Self {
            rank: bytes[8],
            world: bytes[9],
            mesh: read_u64(16),
            hash: read_u64(24),
            collectives: read_u64(32),
            unique_id,
        })
    }
}

fn validate_wire(message: &WireMessage, expected_rank: u8) -> Result<(), NcclTransportError> {
    if message.world as usize != WORLD {
        return Err(NcclTransportError::WorldSize {
            expected: WORLD,
            found: message.world as usize,
        });
    }
    if message.rank != expected_rank {
        return Err(NcclTransportError::RemoteRank {
            expected: expected_rank as usize,
            found: message.rank as usize,
        });
    }
    Ok(())
}

fn write_wire(stream: &mut TcpStream, message: WireMessage) -> Result<(), NcclTransportError> {
    stream
        .write_all(&message.encode())
        .map_err(|error| io_error("write bootstrap", error))
}

fn read_wire(stream: &mut TcpStream) -> Result<WireMessage, NcclTransportError> {
    let mut bytes = [0; WIRE_BYTES];
    stream
        .read_exact(&mut bytes)
        .map_err(|error| io_error("read bootstrap", error))?;
    WireMessage::decode(bytes)
}

fn id_to_bytes(id: &Id) -> [u8; UNIQUE_ID_BYTES] {
    let mut bytes = [0; UNIQUE_ID_BYTES];
    for (destination, source) in bytes.iter_mut().zip(id.internal()) {
        *destination = *source as u8;
    }
    bytes
}

fn id_from_bytes(bytes: [u8; UNIQUE_ID_BYTES]) -> Id {
    let mut internal = [0 as c_char; UNIQUE_ID_BYTES];
    for (destination, source) in internal.iter_mut().zip(bytes) {
        *destination = source as c_char;
    }
    Id::uninit(internal)
}

fn format_cuda_uuid(bytes: [c_char; 16]) -> String {
    use core::fmt::Write as _;

    let mut output = String::with_capacity(36);
    for (index, byte) in bytes.into_iter().enumerate() {
        if matches!(index, 4 | 6 | 8 | 10) {
            output.push('-');
        }
        let _ = write!(output, "{:02x}", byte as u8);
    }
    output
}

#[allow(clippy::too_many_arguments)]
fn launch_by_dtype(
    comm: &Comm,
    stream: &std::sync::Arc<CudaStream>,
    rank: usize,
    kind: CollectiveKind,
    dtype: DTypeId,
    input: &CudaSlice<u8>,
    output: &mut CudaSlice<u8>,
    input_elements: usize,
    output_elements: usize,
) -> Result<(), NcclTransportError> {
    macro_rules! launch {
        ($element:ty) => {
            launch_typed::<$element>(
                comm,
                stream,
                rank,
                kind,
                input,
                output,
                input_elements,
                output_elements,
            )
        };
    }
    match dtype {
        DTypeId::U8 => launch!(u8),
        DTypeId::U32 => launch!(u32),
        DTypeId::I64 => launch!(i64),
        DTypeId::BF16 => launch!(half::bf16),
        DTypeId::F16 => launch!(half::f16),
        DTypeId::F32 => launch!(f32),
        DTypeId::F64 => launch!(f64),
        DTypeId::Q8_0 => Err(CollectiveError::UnsupportedDType { dtype }.into()),
        _ => Err(CollectiveError::UnsupportedDType { dtype }.into()),
    }
}

#[allow(clippy::too_many_arguments)]
fn launch_typed<T: DeviceRepr + NcclType>(
    comm: &Comm,
    stream: &std::sync::Arc<CudaStream>,
    rank: usize,
    kind: CollectiveKind,
    input: &CudaSlice<u8>,
    output: &mut CudaSlice<u8>,
    input_elements: usize,
    output_elements: usize,
) -> Result<(), NcclTransportError> {
    // SAFETY: NcclBuffer checked that the byte allocation is exactly the
    // scalar dtype's checked storage size, and dispatch selected T from that
    // same DTypeId. Q8_0 never reaches this function.
    let input = unsafe { input.transmute::<T>(input_elements) }.ok_or(
        NcclTransportError::Protocol("input reinterpretation failed"),
    )?;
    // SAFETY: output was allocated from the descriptor's checked byte count
    // and T is selected from the descriptor's validated scalar dtype.
    let mut output = unsafe { output.transmute_mut::<T>(output_elements) }.ok_or(
        NcclTransportError::Protocol("output reinterpretation failed"),
    )?;
    match kind {
        CollectiveKind::AllReduce(op) => {
            comm.all_reduce(&input, &mut output, &nccl_reduce(op))
                .map_err(nccl_error)?;
        }
        CollectiveKind::AllGather => {
            comm.all_gather(&input, &mut output).map_err(nccl_error)?;
        }
        CollectiveKind::ReduceScatter(op) => {
            comm.reduce_scatter(&input, &mut output, &nccl_reduce(op))
                .map_err(nccl_error)?;
        }
        CollectiveKind::AllToAll => {
            if input_elements != output_elements || !input_elements.is_multiple_of(WORLD) {
                return Err(CollectiveError::NonDivisible {
                    elements: input_elements,
                    ranks: WORLD,
                }
                .into());
            }
            let chunk = input_elements / WORLD;
            let (input0, input1) = input.split_at(chunk);
            let (mut output0, mut output1) = output.split_at_mut(chunk);
            catch_nccl_panic("launch all-to-all group", || {
                match rank {
                    0 => {
                        stream
                            .memcpy_dtod(&input0, &mut output0)
                            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
                        let mut group = comm.group();
                        group.send(input1, 1).map_err(nccl_error)?;
                        group.recv(output1, 1).map_err(nccl_error)?;
                    }
                    1 => {
                        stream
                            .memcpy_dtod(&input1, &mut output1)
                            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
                        let mut group = comm.group();
                        group.send(input0, 0).map_err(nccl_error)?;
                        group.recv(output0, 0).map_err(nccl_error)?;
                    }
                    found => {
                        return Err(NcclTransportError::RemoteRank { expected: 1, found });
                    }
                }
                Ok(())
            })??;
        }
        CollectiveKind::SendRecv {
            source,
            destination,
        } => {
            if source >= WORLD {
                return Err(CollectiveError::PeerOutOfRange {
                    endpoint: "source",
                    rank: source,
                    ranks: WORLD,
                }
                .into());
            }
            if destination >= WORLD {
                return Err(CollectiveError::PeerOutOfRange {
                    endpoint: "destination",
                    rank: destination,
                    ranks: WORLD,
                }
                .into());
            }
            if source == destination {
                return Err(CollectiveError::SamePeer { rank: source }.into());
            }
            if input_elements != output_elements {
                return Err(NcclTransportError::Elements {
                    expected: input_elements,
                    found: output_elements,
                });
            }
            match rank {
                rank if rank == source => {
                    stream
                        .memcpy_dtod(&input, &mut output)
                        .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
                    comm.send(&input, destination as i32).map_err(nccl_error)?;
                }
                rank if rank == destination => {
                    comm.recv(&mut output, source as i32).map_err(nccl_error)?;
                }
                found => {
                    return Err(NcclTransportError::LocalRank {
                        rank: found,
                        world: WORLD,
                    });
                }
            }
        }
        _ => return Err(NcclTransportError::UnsupportedCollective),
    }
    Ok(())
}

fn validate_reduction(kind: CollectiveKind, dtype: DTypeId) -> Result<(), NcclTransportError> {
    let op = match kind {
        CollectiveKind::AllReduce(op) | CollectiveKind::ReduceScatter(op) => op,
        CollectiveKind::AllGather | CollectiveKind::AllToAll | CollectiveKind::SendRecv { .. } => {
            return Ok(());
        }
        _ => return Err(NcclTransportError::UnsupportedCollective),
    };
    incin_core::dist::validate_collective_reduction(dtype, op).map_err(Into::into)
}

const fn nccl_reduce(op: ReduceOp) -> cudarc::nccl::ReduceOp {
    match op {
        ReduceOp::Sum => cudarc::nccl::ReduceOp::Sum,
        ReduceOp::Mean => cudarc::nccl::ReduceOp::Avg,
        ReduceOp::Max => cudarc::nccl::ReduceOp::Max,
        ReduceOp::Min => cudarc::nccl::ReduceOp::Min,
        ReduceOp::Prod => cudarc::nccl::ReduceOp::Prod,
    }
}

fn io_error(operation: &'static str, error: io::Error) -> NcclTransportError {
    if matches!(
        error.kind(),
        io::ErrorKind::TimedOut | io::ErrorKind::WouldBlock
    ) {
        NcclTransportError::IoTimeout { operation }
    } else {
        NcclTransportError::Io {
            operation,
            kind: error.kind(),
            message: error.to_string(),
        }
    }
}

fn nccl_error(error: cudarc::nccl::result::NcclError) -> NcclTransportError {
    NcclTransportError::Nccl(format!("{:?}", error.0))
}

fn catch_nccl_panic<T>(
    operation: &'static str,
    call: impl FnOnce() -> T,
) -> Result<T, NcclTransportError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(call)).map_err(|payload| {
        let message = payload
            .downcast_ref::<&str>()
            .map_or_else(
                || {
                    payload
                        .downcast_ref::<String>()
                        .map_or("unknown loader panic", String::as_str)
                },
                |message| *message,
            )
            .to_string();
        NcclTransportError::NcclUnavailable { operation, message }
    })
}

/// Structured bootstrap, plan, metadata, CUDA, and NCCL failures.
#[non_exhaustive]
#[derive(thiserror::Error, Debug)]
pub enum NcclTransportError {
    /// The shared process context is shut down or failed.
    #[error(transparent)]
    Context(#[from] ContextError),
    /// Shared collective validation failed.
    #[error(transparent)]
    Collective(#[from] CollectiveError),
    /// Cross-rank plan agreement failed.
    #[error(transparent)]
    Plan(#[from] PlanError),
    /// Data-parallel dtype or gradient-plan validation failed.
    #[error(transparent)]
    DataParallel(#[from] DataParallelError),
    /// Tensor-parallel shape, dtype, or plan validation failed.
    #[error(transparent)]
    TensorParallel(#[from] TensorParallelError),
    /// Pipeline shape, dtype, schedule, or plan validation failed.
    #[error(transparent)]
    Pipeline(#[from] PipelineError),
    /// The requested parameter has no gradient in the completed backward pass.
    #[error("parameter gradient {id:?} was not produced by backward")]
    MissingGradient { id: GradientId },
    /// Caller and plan disagree on which parameter occupies this sequence.
    #[error("gradient identity is {found}, expected {expected}")]
    GradientIdentity { expected: u64, found: u64 },
    /// The next descriptor is not a DP mean all-reduce.
    #[error(
        "gradient descriptor must be Partial<Mean> -> Replicated all-reduce, got {kind:?} from {from_placement:?} to {to_placement:?}"
    )]
    NotDataParallelGradient {
        kind: CollectiveKind,
        from_placement: incin_core::dist::PlacementKind,
        to_placement: incin_core::dist::PlacementKind,
    },
    /// A gradient allocation belongs to another local CUDA ordinal.
    #[error("gradient is on CUDA device {found}, transport owns CUDA device {expected}")]
    GradientDevice { expected: usize, found: usize },
    /// NCCL's scalar buffer path requires a contiguous offset-zero gradient.
    #[error(
        "gradient layout is {layout:?} at element offset {offset}; contiguous offset-zero required"
    )]
    GradientLayout {
        offset: usize,
        layout: incin_core::exec::LayoutClass,
    },
    /// Caller and plan disagree on the tensor-parallel semantic operation.
    #[error("tensor-parallel operation tag is {found}, expected {expected}")]
    TensorParallelIdentity { expected: u64, found: u64 },
    /// The next descriptor does not implement the requested TP operation.
    #[error(
        "tensor-parallel descriptor for {expected:?} got {kind:?} from {from_placement:?} to {to_placement:?}"
    )]
    NotTensorParallelOperation {
        expected: TensorParallelCollective,
        kind: CollectiveKind,
        from_placement: incin_core::dist::PlacementKind,
        to_placement: incin_core::dist::PlacementKind,
    },
    /// A TP tensor allocation belongs to another local CUDA ordinal.
    #[error(
        "tensor-parallel input is on CUDA device {found}, transport owns CUDA device {expected}"
    )]
    TensorParallelDevice { expected: usize, found: usize },
    /// Direct NCCL tensor execution requires contiguous offset-zero storage.
    #[error(
        "tensor-parallel input layout is {layout:?} at element offset {offset}; contiguous offset-zero required"
    )]
    TensorParallelLayout {
        offset: usize,
        layout: incin_core::exec::LayoutClass,
    },
    /// Rank-local tensor shape disagrees with the requested logical shard.
    #[error("tensor-parallel input shape is {found:?}, expected {expected:?}")]
    TensorParallelShape {
        expected: Vec<usize>,
        found: Vec<usize>,
    },
    /// Requested global shape disagrees with the immutable plan.
    #[error("tensor-parallel output has {found} elements, expected {expected}")]
    TensorParallelOutputElements { expected: usize, found: usize },
    /// Caller and plan disagree on pipeline boundary, direction, or microbatch.
    #[error("pipeline transfer tag is {found}, expected {expected}")]
    PipelineIdentity { expected: u64, found: u64 },
    /// The next descriptor does not implement the requested pipeline transfer.
    #[error(
        "pipeline descriptor for {expected:?} got {kind:?} from {from_placement:?} to {to_placement:?}"
    )]
    NotPipelineTransfer {
        expected: PipelineTransfer,
        kind: CollectiveKind,
        from_placement: incin_core::dist::PlacementKind,
        to_placement: incin_core::dist::PlacementKind,
    },
    /// A pipeline tensor allocation belongs to another local CUDA ordinal.
    #[error("pipeline input is on CUDA device {found}, transport owns CUDA device {expected}")]
    PipelineDevice { expected: usize, found: usize },
    /// Direct pipeline transfer requires contiguous offset-zero storage.
    #[error(
        "pipeline input layout is {layout:?} at element offset {offset}; contiguous offset-zero required"
    )]
    PipelineLayout {
        offset: usize,
        layout: incin_core::exec::LayoutClass,
    },
    /// A checked buffer could not be represented.
    #[error("invalid NCCL buffer: {0}")]
    InvalidBuffer(String),
    /// A CUDA allocation has a different byte count.
    #[error("NCCL buffer has {found} bytes, expected {expected}")]
    BufferBytes { expected: usize, found: usize },
    /// Runtime dtype differs from the next descriptor.
    #[error("NCCL buffer has dtype {found:?}, expected {expected:?}")]
    DType { expected: DTypeId, found: DTypeId },
    /// Runtime element count differs from the next descriptor.
    #[error("NCCL buffer has {found} elements, expected {expected}")]
    Elements { expected: usize, found: usize },
    /// The immutable plan has no next launch.
    #[error("collective plan is exhausted after {collectives} launches")]
    PlanExhausted { collectives: usize },
    /// Descriptor order is not the canonical zero-based sequence.
    #[error("collective sequence is {found}, expected {expected}")]
    Sequence { expected: u64, found: u64 },
    /// This transport is deliberately fixed at two network ranks.
    #[error("NCCL group has {found} ranks, expected {expected}")]
    GroupCardinality { expected: usize, found: usize },
    /// A newer collective kind is not implemented by this transport version.
    #[error("collective kind is unsupported by this NCCL transport version")]
    UnsupportedCollective,
    /// Bootstrap peer announced another world size.
    #[error("remote bootstrap world is {found}, expected {expected}")]
    WorldSize { expected: usize, found: usize },
    /// Bootstrap peer announced the wrong rank.
    #[error("remote bootstrap rank is {found}, expected {expected}")]
    RemoteRank { expected: usize, found: usize },
    /// This process's configured rank is outside the two-rank world.
    #[error("local rank {rank} is outside a world of {world}")]
    LocalRank { rank: usize, world: usize },
    /// Rank zero did not supply its NCCL id to the protocol.
    #[error("rank zero bootstrap requires an NCCL unique id")]
    MissingRootId,
    /// Rank one attempted to choose the communicator identity.
    #[error("rank one must receive, not supply, the NCCL unique id")]
    UnexpectedPeerId,
    /// Wire data did not follow the versioned protocol.
    #[error("invalid NCCL bootstrap protocol: {0}")]
    Protocol(&'static str),
    /// A fixed-size topology field could not carry its value.
    #[error("{field} has {found} bytes, maximum is {maximum}")]
    FieldTooLong {
        field: &'static str,
        maximum: usize,
        found: usize,
    },
    /// Hosts loaded different transport implementations or versions.
    #[error("transport mismatch: local {local}, remote {remote}")]
    TransportMismatch { local: String, remote: String },
    /// Zero or overflowing durations cannot form a deadline.
    #[error("NCCL timeout must be positive and fit Instant")]
    InvalidTimeout,
    /// A bounded phase reached its deadline.
    #[error("{phase} timed out after {timeout:?}")]
    Timeout {
        phase: &'static str,
        timeout: Duration,
    },
    /// Socket I/O reported a timeout.
    #[error("{operation} timed out")]
    IoTimeout { operation: &'static str },
    /// Other socket failure.
    #[error("{operation} failed ({kind:?}): {message}")]
    Io {
        operation: &'static str,
        kind: io::ErrorKind,
        message: String,
    },
    /// CUDA driver failure.
    #[error("CUDA failure: {0}")]
    Cuda(String),
    /// NCCL failure.
    #[error("NCCL failure: {0}")]
    Nccl(String),
    /// cudarc could not dynamically load the NCCL shared library.
    #[error("cannot {operation}: NCCL is unavailable ({message})")]
    NcclUnavailable {
        operation: &'static str,
        message: String,
    },
    /// NCCL returned a negative or otherwise unrepresentable version.
    #[error("NCCL returned invalid encoded version {0}")]
    InvalidNcclVersion(i32),
}

#[cfg(test)]
mod tests {
    use super::*;
    use incin_core::backend_authoring::HostInterop;

    fn all_reduce_plan() -> CollectivePlan {
        type Mesh =
            incin_core::dist::mesh::MeshSpec<incin_core::dist::mesh::Data<incin_core::typenum::U2>>;
        type PartialSum = incin_core::dist::Partial<Mesh, incin_core::dist::Sum>;
        type Replica = incin_core::dist::Replicated<Mesh>;
        let identities = [
            DeviceIdentity::new(DeviceId::cuda(0), "GPU-a".into(), "sm_90".into()),
            DeviceIdentity::new(DeviceId::cuda(1), "GPU-b".into(), "sm_90".into()),
        ];
        let topology = NcclTopology::new(
            identities,
            0,
            TransportVersion::new("nccl".into(), 2, 30, 0),
        )
        .unwrap();
        let mesh = incin_core::dist::mesh::DeviceMesh::<Mesh>::bind(
            &[DeviceId::cuda(0), DeviceId::cuda(1)],
            &topology,
        )
        .unwrap();
        let mut builder = incin_core::dist::CollectivePlanBuilder::new(&mesh);
        builder
            .push_static_tagged::<f32, PartialSum, Replica>(
                incin_core::dist::CollectiveTag::new(41),
                incin_core::dist::mesh::MeshAxis::Data,
                0,
                4,
                StreamId::default(),
                None,
            )
            .unwrap();
        builder.finish()
    }

    fn data_parallel_plan() -> CollectivePlan {
        let identities = [
            DeviceIdentity::new(DeviceId::cuda(0), "GPU-a".into(), "sm_90".into()),
            DeviceIdentity::new(DeviceId::cuda(1), "GPU-b".into(), "sm_90".into()),
        ];
        let topology = NcclTopology::new(
            identities,
            0,
            TransportVersion::new("nccl".into(), 2, 30, 0),
        )
        .unwrap();
        let mesh =
            incin_core::dist::mesh::DeviceMesh::<incin_core::dist::TwoRankDataParallel>::bind(
                &[DeviceId::cuda(0), DeviceId::cuda(1)],
                &topology,
            )
            .unwrap();
        let mut builder = incin_core::dist::DataParallelPlanBuilder::new(&mesh, 0);
        builder
            .push_static::<f32>(GradientId::new(41).unwrap(), 4, StreamId::default())
            .unwrap();
        builder.finish().unwrap().into_collective_plan()
    }

    fn tensor_parallel_plan() -> CollectivePlan {
        let identities = [
            DeviceIdentity::new(DeviceId::cuda(0), "GPU-a".into(), "sm_90".into()),
            DeviceIdentity::new(DeviceId::cuda(1), "GPU-b".into(), "sm_90".into()),
        ];
        let topology = NcclTopology::new(
            identities,
            0,
            TransportVersion::new("nccl".into(), 2, 30, 0),
        )
        .unwrap();
        let mesh =
            incin_core::dist::mesh::DeviceMesh::<incin_core::dist::TwoRankTensorParallel>::bind(
                &[DeviceId::cuda(0), DeviceId::cuda(1)],
                &topology,
            )
            .unwrap();
        let mut builder = incin_core::dist::TensorParallelPlanBuilder::new(&mesh, 0);
        builder
            .push_column_static::<f32, incin_core::typenum::U0, incin_core::typenum::U4>(
                TensorParallelId::new(51).unwrap(),
                1,
                StreamId::default(),
            )
            .unwrap();
        builder
            .push_row_static::<f32, incin_core::typenum::U4>(
                TensorParallelId::new(52).unwrap(),
                2,
                StreamId::default(),
            )
            .unwrap();
        builder.finish().unwrap().into_collective_plan()
    }

    fn pipeline_plan() -> CollectivePlan {
        let identities = [
            DeviceIdentity::new(DeviceId::cuda(0), "GPU-a".into(), "sm_90".into()),
            DeviceIdentity::new(DeviceId::cuda(1), "GPU-b".into(), "sm_90".into()),
        ];
        let topology = NcclTopology::new(
            identities,
            0,
            TransportVersion::new("nccl".into(), 2, 30, 0),
        )
        .unwrap();
        let mesh = incin_core::dist::mesh::DeviceMesh::<incin_core::dist::TwoRankPipeline>::bind(
            &[DeviceId::cuda(0), DeviceId::cuda(1)],
            &topology,
        )
        .unwrap();
        incin_core::dist::PipelinePlanBuilder::build_static::<
            f32,
            incin_core::shapes::DimCons<
                incin_core::shapes::Static<incin_core::typenum::U2>,
                incin_core::shapes::Nil,
            >,
            incin_core::typenum::U2,
            incin_core::dist::GPipe,
        >(
            &mesh,
            0,
            PipelineBoundaryId::new(61).unwrap(),
            incin_core::dist::ActivationCheckpoint::Keep,
            StreamId::default(),
        )
        .unwrap()
        .into_collective_plan()
    }

    fn summary(mesh: u64, hash: u64, collectives: usize) -> PlanSummary {
        PlanSummary::from_parts(MeshId::from_digest(mesh), hash, collectives)
    }

    fn localhost_listener() -> (TcpListener, SocketAddr) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        (listener, address)
    }

    #[test]
    fn wire_round_trip_is_fixed_size_and_preserves_identity() {
        let id = core::array::from_fn(|index| index as u8);
        let message = WireMessage::new(1, summary(7, 11, 13), id);
        let encoded = message.encode();
        assert_eq!(encoded.len(), WIRE_BYTES);
        let decoded = WireMessage::decode(encoded).unwrap();
        assert_eq!(decoded.rank, 1);
        assert_eq!(decoded.summary().unwrap(), summary(7, 11, 13));
        assert_eq!(decoded.unique_id, id);
    }

    #[test]
    fn topology_is_networked_rank_local_and_stable_across_processes() {
        let identities = [
            DeviceIdentity::new(DeviceId::cuda(0), "GPU-a".into(), "sm_90".into()),
            DeviceIdentity::new(DeviceId::cuda(1), "GPU-b".into(), "sm_90".into()),
        ];
        let version = TransportVersion::new("nccl".into(), 2, 30, 0);
        let rank0 = NcclTopology::new(identities.clone(), 0, version.clone()).unwrap();
        let rank1 = NcclTopology::new(identities, 1, version).unwrap();
        assert_eq!(
            rank0.link(DeviceId::cuda(0), DeviceId::cuda(1)),
            LinkClass::Network
        );
        assert_eq!(rank0.transport(), rank1.transport());
        assert_eq!(
            rank0.layout(),
            ProcessLayout::ProcessPerRank { rank: 0, world: 2 }
        );
        assert_eq!(
            rank1.layout(),
            ProcessLayout::ProcessPerRank { rank: 1, world: 2 }
        );
    }

    #[test]
    fn topology_wire_round_trip_preserves_identity_and_version() {
        let identity = DeviceIdentity::new(DeviceId::cuda(1), "GPU-abc".into(), "sm_90".into());
        let transport = TransportVersion::new("nccl".into(), 2, 30, 1);
        let wire = TopologyWire::new(1, &identity, &transport).unwrap();
        let decoded = TopologyWire::decode(wire.encode()).unwrap();
        assert_eq!(decoded.identity().unwrap(), identity);
        assert_eq!(decoded.transport().unwrap(), transport);
    }

    #[test]
    fn two_tcp_ranks_discover_the_same_ordered_topology() {
        let (reserved, address) = localhost_listener();
        drop(reserved);
        let timeout = Duration::from_secs(2);
        let version = TransportVersion::new("nccl".into(), 2, 30, 0);
        let root_identity =
            DeviceIdentity::new(DeviceId::cuda(0), "GPU-root".into(), "sm_90".into());
        let peer_identity =
            DeviceIdentity::new(DeviceId::cuda(1), "GPU-peer".into(), "sm_90".into());
        let root_version = version.clone();
        let root = thread::spawn(move || {
            exchange_topology(
                TwoRankBootstrapConfig::root(address, timeout),
                root_identity,
                root_version,
            )
        });
        let peer = exchange_topology(
            TwoRankBootstrapConfig::peer(address, timeout),
            peer_identity,
            version,
        )
        .unwrap();
        let root = root.join().unwrap().unwrap();
        for device in [DeviceId::cuda(0), DeviceId::cuda(1)] {
            assert_eq!(root.identify(device), peer.identify(device));
        }
        assert_eq!(root.transport(), peer.transport());
    }

    #[test]
    fn discovery_then_plan_bootstrap_reuses_one_config() {
        // Repeat enough times to exercise the listener handoff race: the peer
        // begins phase two as soon as its phase-one reply arrives.
        for _ in 0..16 {
            let (reserved, address) = localhost_listener();
            drop(reserved);
            let timeout = Duration::from_secs(2);
            let local_summary = summary(17, 19, 2);
            let root = thread::spawn(move || {
                let topology = exchange_topology(
                    TwoRankBootstrapConfig::root(address, timeout),
                    DeviceIdentity::new(DeviceId::cuda(0), "GPU-root".into(), "sm_90".into()),
                    TransportVersion::new("nccl".into(), 2, 30, 0),
                )?;
                let bootstrap = exchange_bootstrap(
                    TwoRankBootstrapConfig::root(address, timeout),
                    local_summary,
                    Some([42; UNIQUE_ID_BYTES]),
                )?;
                Ok::<_, NcclTransportError>((topology, bootstrap))
            });
            let peer_topology = exchange_topology(
                TwoRankBootstrapConfig::peer(address, timeout),
                DeviceIdentity::new(DeviceId::cuda(1), "GPU-peer".into(), "sm_90".into()),
                TransportVersion::new("nccl".into(), 2, 30, 0),
            )
            .unwrap();
            let peer_bootstrap = exchange_bootstrap(
                TwoRankBootstrapConfig::peer(address, timeout),
                local_summary,
                None,
            )
            .unwrap();
            let (root_topology, root_bootstrap) = root.join().unwrap().unwrap();
            assert_eq!(
                root_topology.identify(DeviceId::cuda(1)),
                peer_topology.identify(DeviceId::cuda(1))
            );
            assert_eq!(root_bootstrap.agreed, peer_bootstrap.agreed);
            assert_eq!(root_bootstrap.unique_id, peer_bootstrap.unique_id);
        }
    }

    #[test]
    fn cuda_uuid_format_is_canonical_and_unsigned() {
        let bytes: [c_char; 16] = core::array::from_fn(|index| (0xf0 + index as u8) as c_char);
        assert_eq!(
            format_cuda_uuid(bytes),
            "f0f1f2f3-f4f5-f6f7-f8f9-fafbfcfdfeff"
        );
    }

    #[test]
    #[ignore = "requires one CUDA device"]
    fn local_cuda_identity_uses_uuid_and_compute_capability() {
        let identity = NcclTopology::probe_local_cuda_identity(0, 0).unwrap();
        assert_eq!(identity.device(), DeviceId::cuda(0));
        assert_eq!(identity.persistent().len(), 36);
        assert!(identity.architecture().starts_with("sm_"));
    }

    #[test]
    fn corrupt_magic_and_wrong_identity_are_structured_failures() {
        let mut encoded = WireMessage::new(1, summary(7, 11, 13), [0; 128]).encode();
        encoded[0] ^= 1;
        assert!(matches!(
            WireMessage::decode(encoded),
            Err(NcclTransportError::Protocol(_))
        ));
        assert!(matches!(
            validate_wire(&WireMessage::new(0, summary(7, 11, 13), [0; 128]), 1),
            Err(NcclTransportError::RemoteRank {
                expected: 1,
                found: 0
            })
        ));
    }

    #[test]
    fn two_tcp_ranks_exchange_one_id_and_agree_on_plan() {
        let (reserved, address) = localhost_listener();
        drop(reserved);
        let timeout = Duration::from_secs(2);
        let local = summary(17, 19, 3);
        let id = [23; UNIQUE_ID_BYTES];
        let root = thread::spawn(move || {
            exchange_bootstrap(
                TwoRankBootstrapConfig::root(address, timeout),
                local,
                Some(id),
            )
        });
        let peer = exchange_bootstrap(TwoRankBootstrapConfig::peer(address, timeout), local, None)
            .unwrap();
        let root = root.join().unwrap().unwrap();
        assert_eq!(root.unique_id, id);
        assert_eq!(peer.unique_id, id);
        assert_eq!(root.agreed, peer.agreed);
        assert_eq!(root.agreed.ranks(), WORLD);
    }

    #[test]
    fn divergent_plan_is_rejected_before_communicator_creation() {
        let (reserved, address) = localhost_listener();
        drop(reserved);
        let timeout = Duration::from_secs(2);
        let root = thread::spawn(move || {
            exchange_bootstrap(
                TwoRankBootstrapConfig::root(address, timeout),
                summary(17, 19, 3),
                Some([23; UNIQUE_ID_BYTES]),
            )
        });
        let peer = exchange_bootstrap(
            TwoRankBootstrapConfig::peer(address, timeout),
            summary(17, 20, 3),
            None,
        );
        let root = root.join().unwrap();
        assert!(matches!(
            root,
            Err(NcclTransportError::Plan(PlanError::PlanHashMismatch {
                rank: 1,
                expected: 19,
                found: 20
            }))
        ));
        assert!(peer.is_err());
    }

    #[test]
    fn missing_peer_hits_a_bounded_accept_timeout() {
        let (reserved, address) = localhost_listener();
        drop(reserved);
        let timeout = Duration::from_millis(15);
        let error = exchange_bootstrap(
            TwoRankBootstrapConfig::root(address, timeout),
            summary(1, 2, 0),
            Some([0; UNIQUE_ID_BYTES]),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            NcclTransportError::Timeout {
                phase: "accept rank one",
                ..
            }
        ));
    }

    #[test]
    fn dyn_dtype_validation_matches_static_collective_policy() {
        assert!(validate_collective_dtype(DTypeId::F64).is_ok());
        assert_eq!(
            validate_collective_dtype(DTypeId::Q8_0).unwrap_err(),
            CollectiveError::UnsupportedDType {
                dtype: DTypeId::Q8_0
            }
        );
        assert!(matches!(
            validate_reduction(CollectiveKind::AllReduce(ReduceOp::Mean), DTypeId::U32),
            Err(NcclTransportError::Collective(
                CollectiveError::UnsupportedReduction {
                    dtype: DTypeId::U32,
                    op: ReduceOp::Mean
                }
            ))
        ));
    }

    #[test]
    fn dynamic_loader_panics_become_structured_errors() {
        let error = catch_nccl_panic("load test", || panic!("libnccl not found")).unwrap_err();
        assert!(matches!(
            error,
            NcclTransportError::NcclUnavailable {
                operation: "load test",
                message
            } if message == "libnccl not found"
        ));
    }

    #[test]
    #[ignore = "probes the optional system NCCL shared library"]
    fn installed_nccl_probe_returns_instead_of_unwinding() {
        let _available_or_structured_error = NcclTopology::installed_transport_version();
    }

    #[test]
    fn launch_preflight_rejects_order_dtype_count_and_byte_drift() {
        let plan = all_reduce_plan();
        let descriptor = &plan.descriptors()[0];
        assert!(validate_launch(descriptor, 0, DTypeId::F32, 4, 16).is_ok());
        assert!(matches!(
            validate_launch(descriptor, 1, DTypeId::F32, 4, 16),
            Err(NcclTransportError::Sequence {
                expected: 1,
                found: 0
            })
        ));
        assert!(matches!(
            validate_launch(descriptor, 0, DTypeId::F64, 4, 32),
            Err(NcclTransportError::DType {
                expected: DTypeId::F32,
                found: DTypeId::F64
            })
        ));
        assert!(matches!(
            validate_launch(descriptor, 0, DTypeId::F32, 3, 12),
            Err(NcclTransportError::Elements {
                expected: 4,
                found: 3
            })
        ));
        assert!(matches!(
            validate_launch(descriptor, 0, DTypeId::F32, 4, 12),
            Err(NcclTransportError::BufferBytes {
                expected: 16,
                found: 12
            })
        ));
        assert!(matches!(
            validate_launch(descriptor, 0, DTypeId::Q8_0, 4, 16),
            Err(NcclTransportError::Collective(
                CollectiveError::UnsupportedDType {
                    dtype: DTypeId::Q8_0
                }
            ))
        ));
    }

    #[test]
    fn gradient_preflight_requires_identity_mean_placement_and_dyn_float_dtype() {
        let plan = data_parallel_plan();
        let descriptor = &plan.descriptors()[0];
        assert!(
            validate_gradient_launch(
                descriptor,
                0,
                GradientId::new(41).unwrap(),
                DTypeId::F32,
                4,
                16,
            )
            .is_ok()
        );
        assert!(matches!(
            validate_gradient_launch(
                descriptor,
                0,
                GradientId::new(42).unwrap(),
                DTypeId::F32,
                4,
                16,
            ),
            Err(NcclTransportError::GradientIdentity {
                expected: 41,
                found: 42
            })
        ));
        assert!(matches!(
            validate_gradient_launch(
                descriptor,
                0,
                GradientId::new(41).unwrap(),
                DTypeId::U32,
                4,
                16,
            ),
            Err(NcclTransportError::DataParallel(
                DataParallelError::UnsupportedGradientDType {
                    dtype: DTypeId::U32
                }
            ))
        ));

        let sum = all_reduce_plan();
        assert!(matches!(
            validate_gradient_launch(
                &sum.descriptors()[0],
                0,
                GradientId::new(41).unwrap(),
                DTypeId::F32,
                4,
                16,
            ),
            Err(NcclTransportError::NotDataParallelGradient { .. })
        ));
    }

    #[test]
    fn tensor_parallel_preflight_requires_identity_semantics_and_dyn_float_dtype() {
        let plan = tensor_parallel_plan();
        let column = &plan.descriptors()[0];
        let column_kind = TensorParallelCollective::ColumnOutputGather { tensor_axis: 0 };
        assert!(
            validate_tensor_parallel_launch(
                column,
                0,
                TensorParallelId::new(51).unwrap(),
                column_kind,
                DTypeId::F32,
                2,
                8,
            )
            .is_ok()
        );
        assert!(matches!(
            validate_tensor_parallel_launch(
                column,
                0,
                TensorParallelId::new(52).unwrap(),
                column_kind,
                DTypeId::F32,
                2,
                8,
            ),
            Err(NcclTransportError::TensorParallelIdentity { .. })
        ));
        assert!(matches!(
            validate_tensor_parallel_launch(
                column,
                0,
                TensorParallelId::new(51).unwrap(),
                TensorParallelCollective::AttentionHeadGather { tensor_axis: 0 },
                DTypeId::F32,
                2,
                8,
            ),
            Err(NcclTransportError::TensorParallelIdentity { .. })
        ));
        assert!(matches!(
            validate_tensor_parallel_launch(
                column,
                0,
                TensorParallelId::new(51).unwrap(),
                column_kind,
                DTypeId::U32,
                2,
                8,
            ),
            Err(NcclTransportError::TensorParallel(
                TensorParallelError::UnsupportedTensorDType {
                    dtype: DTypeId::U32
                }
            ))
        ));
        assert_eq!(
            validate_tensor_parallel_shapes(
                column,
                TensorParallelCollective::ColumnOutputGather { tensor_axis: 1 },
                &[2, 1],
                &[2, 2],
            )
            .unwrap(),
            vec![2, 1]
        );
        assert!(matches!(
            validate_tensor_parallel_shapes(
                column,
                TensorParallelCollective::ColumnOutputGather { tensor_axis: 1 },
                &[1, 2],
                &[2, 2],
            ),
            Err(NcclTransportError::TensorParallelShape { .. })
        ));
        assert!(matches!(
            validate_tensor_parallel_shapes(column, column_kind, &[3], &[6]),
            Err(NcclTransportError::TensorParallelOutputElements {
                expected: 4,
                found: 6
            })
        ));

        let row = &plan.descriptors()[1];
        assert!(
            validate_tensor_parallel_launch(
                row,
                1,
                TensorParallelId::new(52).unwrap(),
                TensorParallelCollective::RowOutputSum,
                DTypeId::F32,
                2,
                8,
            )
            .is_ok()
        );
        assert_eq!(
            validate_tensor_parallel_shapes(
                row,
                TensorParallelCollective::RowOutputSum,
                &[1, 2],
                &[1, 2],
            )
            .unwrap(),
            vec![1, 2]
        );
        assert!(matches!(
            validate_tensor_parallel_launch(
                row,
                1,
                TensorParallelId::new(52).unwrap(),
                column_kind,
                DTypeId::F32,
                2,
                8,
            ),
            Err(NcclTransportError::TensorParallelIdentity { .. })
        ));
    }

    #[test]
    fn pipeline_preflight_requires_identity_direction_and_dyn_float_dtype() {
        let plan = pipeline_plan();
        let forward = &plan.descriptors()[0];
        assert!(
            validate_pipeline_launch(
                forward,
                0,
                PipelineBoundaryId::new(61).unwrap(),
                PipelineTransfer::ForwardActivation,
                0,
                DTypeId::F32,
                2,
                8,
            )
            .is_ok()
        );
        assert!(matches!(
            validate_pipeline_launch(
                forward,
                0,
                PipelineBoundaryId::new(62).unwrap(),
                PipelineTransfer::ForwardActivation,
                0,
                DTypeId::F32,
                2,
                8,
            ),
            Err(NcclTransportError::PipelineIdentity { .. })
        ));
        assert!(matches!(
            validate_pipeline_launch(
                forward,
                0,
                PipelineBoundaryId::new(61).unwrap(),
                PipelineTransfer::BackwardGradient,
                0,
                DTypeId::F32,
                2,
                8,
            ),
            Err(NcclTransportError::PipelineIdentity { .. })
        ));
        assert!(matches!(
            validate_pipeline_launch(
                forward,
                0,
                PipelineBoundaryId::new(61).unwrap(),
                PipelineTransfer::ForwardActivation,
                0,
                DTypeId::U32,
                2,
                8,
            ),
            Err(NcclTransportError::Pipeline(
                PipelineError::UnsupportedDType {
                    dtype: DTypeId::U32
                }
            ))
        ));
        assert!(matches!(
            validate_pipeline_launch(
                forward,
                0,
                PipelineBoundaryId::new(61).unwrap(),
                PipelineTransfer::ForwardActivation,
                0,
                DTypeId::F32,
                3,
                12,
            ),
            Err(NcclTransportError::Elements {
                expected: 2,
                found: 3
            })
        ));
    }

    #[test]
    #[ignore = "requires one CUDA device"]
    fn tensor_parallel_reassembly_moves_rank_axis_on_cuda_for_static_and_dyn() {
        type B = CudaBackendImpl<incin_core::tensor::device::CudaN<incin_core::typenum::U0>>;
        type D = incin_core::tensor::device::CudaN<incin_core::typenum::U0>;

        let rank_major = [
            1.0f32, 2.0, 3.0, 2.0, 3.0, 4.0, //
            4.0, 3.0, 7.0, 5.0, 5.0, 9.0,
        ];
        let expected = [
            1.0f32, 2.0, 3.0, 4.0, 3.0, 7.0, //
            2.0, 3.0, 4.0, 5.0, 5.0, 9.0,
        ];
        let collective = TensorParallelCollective::ColumnOutputGather { tensor_axis: 1 };

        let static_input =
            Tensor::<incin_core::shapes::Dyn, B>::from_slice(&rank_major, vec![12]).unwrap();
        let static_output = reassemble_tensor_parallel_storage::<D, f32>(
            static_input.inner(),
            collective,
            &[2, 3],
            &[2, 6],
        )
        .unwrap();
        let static_bytes = B::to_bytes::<f32>(&static_output).unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, f32>(&static_bytes), expected);

        let dyn_input = Tensor::<incin_core::shapes::Dyn, B, incin_core::shapes::Dyn>::from_bytes(
            bytemuck::cast_slice(&rank_major),
            (vec![12], DTypeId::F32),
        )
        .unwrap();
        let dyn_output = reassemble_tensor_parallel_storage::<D, incin_core::shapes::Dyn>(
            dyn_input.inner(),
            collective,
            &[2, 3],
            &[2, 6],
        )
        .unwrap();
        let dyn_bytes = B::to_bytes::<incin_core::shapes::Dyn>(&dyn_output).unwrap();
        assert_eq!(bytemuck::cast_slice::<u8, f32>(&dyn_bytes), expected);
    }
}
