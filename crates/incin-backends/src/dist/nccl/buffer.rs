//! Buffer types and completion events for NCCL transport.

use core::marker::PhantomData;
use std::thread;
use std::time::{Duration, Instant};

use cudarc::driver::{CudaEvent, CudaSlice};
use incin_core::dist::{
    CollectiveKind, DistributedContextHandle, GroupId, SequenceToken, StreamId,
    validate_collective_dtype,
};
use incin_core::shapes::OperationKind;
use incin_core::tensor::dtype::{DType, DTypeId};

use crate::dist::nccl::error::NcclTransportError;

/// Rank-local CUDA bytes indexed by a static dtype or [`Dyn`](incin_core::shapes::Dyn).
#[derive(Debug)]
pub struct NcclBuffer<K: DType> {
    pub(crate) data: CudaSlice<u8>,
    pub(crate) elements: usize,
    pub(crate) dtype: K::Field,
    pub(crate) marker: PhantomData<fn() -> K>,
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
    pub(crate) event: CudaEvent,
    pub(crate) group: GroupId,
    pub(crate) sequence: SequenceToken,
    pub(crate) stream: StreamId,
    pub(crate) kind: CollectiveKind,
    pub(crate) distributed_context: Option<DistributedContextHandle>,
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
