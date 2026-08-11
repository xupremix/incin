//! Deterministic in-process reference collectives.

use core::marker::PhantomData;

use incin_core::dist::mesh::{
    DeviceIdentity, LinkClass, ProcessLayout, TopologyProbe, TransportVersion,
};
use incin_core::exec::ReduceOp;
use incin_core::prelude::{DType, DTypeId, DeviceId};

use super::collective::{
    CollectiveBackend, CollectiveDType, CollectiveError, CollectiveKind, CollectiveOutput, GroupId,
    StreamId,
};

/// Typed value payload used by the reference transport.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum ReferenceValues {
    /// `u8` elements.
    U8(alloc::vec::Vec<u8>),
    /// `u32` elements.
    U32(alloc::vec::Vec<u32>),
    /// `i64` elements.
    I64(alloc::vec::Vec<i64>),
    /// `bf16` elements.
    BF16(alloc::vec::Vec<half::bf16>),
    /// `f16` elements.
    F16(alloc::vec::Vec<half::f16>),
    /// `f32` elements.
    F32(alloc::vec::Vec<f32>),
    /// `f64` elements.
    F64(alloc::vec::Vec<f64>),
}

impl ReferenceValues {
    /// Runtime dtype of this payload.
    #[must_use]
    pub const fn dtype(&self) -> DTypeId {
        match self {
            Self::U8(_) => DTypeId::U8,
            Self::U32(_) => DTypeId::U32,
            Self::I64(_) => DTypeId::I64,
            Self::BF16(_) => DTypeId::BF16,
            Self::F16(_) => DTypeId::F16,
            Self::F32(_) => DTypeId::F32,
            Self::F64(_) => DTypeId::F64,
        }
    }

    /// Logical element count.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::U8(values) => values.len(),
            Self::U32(values) => values.len(),
            Self::I64(values) => values.len(),
            Self::BF16(values) => values.len(),
            Self::F16(values) => values.len(),
            Self::F32(values) => values.len(),
            Self::F64(values) => values.len(),
        }
    }

    /// Whether the payload is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// One rank-local reference buffer, indexed by static or dynamic dtype `K`.
pub struct ReferenceBuffer<K: DType> {
    values: ReferenceValues,
    dtype: K::Field,
    marker: PhantomData<fn() -> K>,
}

impl<K: DType> ReferenceBuffer<K> {
    /// Join values to a static/runtime dtype field.
    pub fn try_new(values: ReferenceValues, dtype: K::Field) -> Result<Self, CollectiveError> {
        let typed =
            K::descriptor(&dtype)
                .builtin_id()
                .ok_or_else(|| CollectiveError::BufferDType {
                    values: values.dtype(),
                    typed: DTypeId::F32,
                })?;
        if values.dtype() != typed {
            return Err(CollectiveError::BufferDType {
                values: values.dtype(),
                typed,
            });
        }
        Ok(Self {
            values,
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

    /// Typed values.
    #[must_use]
    pub const fn values(&self) -> &ReferenceValues {
        &self.values
    }
}

impl<K: DType> Clone for ReferenceBuffer<K> {
    fn clone(&self) -> Self {
        Self {
            values: self.values.clone(),
            dtype: self.dtype.clone(),
            marker: PhantomData,
        }
    }
}

impl<K: DType> core::fmt::Debug for ReferenceBuffer<K> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ReferenceBuffer")
            .field("dtype", &self.dtype())
            .field("values", &self.values)
            .finish()
    }
}

impl<K: DType> PartialEq for ReferenceBuffer<K> {
    fn eq(&self, other: &Self) -> bool {
        self.dtype() == other.dtype() && self.values == other.values
    }
}

/// Synchronously completed reference event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReferenceEvent {
    group: GroupId,
    stream: StreamId,
    kind: CollectiveKind,
}

impl ReferenceEvent {
    /// Group that completed.
    #[must_use]
    pub const fn group(self) -> GroupId {
        self.group
    }

    /// Stream dependency attached to the operation.
    #[must_use]
    pub const fn stream(self) -> StreamId {
        self.stream
    }

    /// Collective that completed.
    #[must_use]
    pub const fn kind(self) -> CollectiveKind {
        self.kind
    }
}

/// Stateless deterministic reference transport.
#[derive(Debug, Default, Clone, Copy)]
pub struct ReferenceTransport;

impl CollectiveBackend for ReferenceTransport {
    type Buffer<K: DType> = ReferenceBuffer<K>;
    type Event = ReferenceEvent;

    fn all_reduce<K: CollectiveDType>(
        &self,
        group: GroupId,
        inputs: &[Self::Buffer<K>],
        op: ReduceOp,
        stream: StreamId,
    ) -> Result<CollectiveOutput<Self::Buffer<K>, Self::Event>, CollectiveError> {
        validate_inputs(group, inputs)?;
        let reduced = reduce(inputs, op)?;
        Ok(CollectiveOutput::new(
            alloc::vec![reduced; group.ranks()],
            ReferenceEvent {
                group,
                stream,
                kind: CollectiveKind::AllReduce(op),
            },
        ))
    }

    fn all_gather<K: CollectiveDType>(
        &self,
        group: GroupId,
        inputs: &[Self::Buffer<K>],
        stream: StreamId,
    ) -> Result<CollectiveOutput<Self::Buffer<K>, Self::Event>, CollectiveError> {
        validate_inputs(group, inputs)?;
        let values = concatenate(inputs)?;
        let gathered = ReferenceBuffer::try_new(values, inputs[0].dtype.clone())?;
        Ok(CollectiveOutput::new(
            alloc::vec![gathered; group.ranks()],
            ReferenceEvent {
                group,
                stream,
                kind: CollectiveKind::AllGather,
            },
        ))
    }

    fn reduce_scatter<K: CollectiveDType>(
        &self,
        group: GroupId,
        inputs: &[Self::Buffer<K>],
        op: ReduceOp,
        stream: StreamId,
    ) -> Result<CollectiveOutput<Self::Buffer<K>, Self::Event>, CollectiveError> {
        validate_inputs(group, inputs)?;
        let reduced = reduce(inputs, op)?;
        let chunks = split(reduced.values, group.ranks())?;
        let mut buffers = alloc::vec::Vec::with_capacity(group.ranks());
        for values in chunks {
            buffers.push(ReferenceBuffer::try_new(values, reduced.dtype.clone())?);
        }
        Ok(CollectiveOutput::new(
            buffers,
            ReferenceEvent {
                group,
                stream,
                kind: CollectiveKind::ReduceScatter(op),
            },
        ))
    }

    fn all_to_all<K: CollectiveDType>(
        &self,
        group: GroupId,
        inputs: &[Self::Buffer<K>],
        stream: StreamId,
    ) -> Result<CollectiveOutput<Self::Buffer<K>, Self::Event>, CollectiveError> {
        validate_inputs(group, inputs)?;
        let mut source_chunks = alloc::vec::Vec::with_capacity(group.ranks());
        for input in inputs {
            source_chunks.push(split(input.values.clone(), group.ranks())?);
        }
        let mut buffers = alloc::vec::Vec::with_capacity(group.ranks());
        for destination in 0..group.ranks() {
            let chunks: alloc::vec::Vec<&ReferenceValues> = source_chunks
                .iter()
                .map(|source| &source[destination])
                .collect();
            let values = concatenate_values(&chunks)?;
            buffers.push(ReferenceBuffer::try_new(values, inputs[0].dtype.clone())?);
        }
        Ok(CollectiveOutput::new(
            buffers,
            ReferenceEvent {
                group,
                stream,
                kind: CollectiveKind::AllToAll,
            },
        ))
    }

    fn send_recv<K: CollectiveDType>(
        &self,
        group: GroupId,
        inputs: &[Self::Buffer<K>],
        source: usize,
        destination: usize,
        stream: StreamId,
    ) -> Result<CollectiveOutput<Self::Buffer<K>, Self::Event>, CollectiveError> {
        validate_inputs(group, inputs)?;
        validate_peer("source", source, group.ranks())?;
        validate_peer("destination", destination, group.ranks())?;
        if source == destination {
            return Err(CollectiveError::SamePeer { rank: source });
        }
        let mut buffers = inputs.to_vec();
        buffers[destination] = inputs[source].clone();
        Ok(CollectiveOutput::new(
            buffers,
            ReferenceEvent {
                group,
                stream,
                kind: CollectiveKind::SendRecv {
                    source,
                    destination,
                },
            },
        ))
    }
}

fn validate_peer(endpoint: &'static str, rank: usize, ranks: usize) -> Result<(), CollectiveError> {
    if rank >= ranks {
        Err(CollectiveError::PeerOutOfRange {
            endpoint,
            rank,
            ranks,
        })
    } else {
        Ok(())
    }
}

fn validate_inputs<K: DType>(
    group: GroupId,
    inputs: &[ReferenceBuffer<K>],
) -> Result<(), CollectiveError> {
    if inputs.len() != group.ranks() {
        return Err(CollectiveError::InputCount {
            expected: group.ranks(),
            found: inputs.len(),
        });
    }
    let expected_dtype = inputs[0].dtype();
    let expected_len = inputs[0].values.len();
    for (rank, input) in inputs.iter().enumerate().skip(1) {
        if input.dtype() != expected_dtype {
            return Err(CollectiveError::DTypeMismatch {
                rank,
                expected: expected_dtype,
                found: input.dtype(),
            });
        }
        if input.values.len() != expected_len {
            return Err(CollectiveError::ElementCount {
                rank,
                expected: expected_len,
                found: input.values.len(),
            });
        }
    }
    Ok(())
}

fn reduce<K: DType>(
    inputs: &[ReferenceBuffer<K>],
    op: ReduceOp,
) -> Result<ReferenceBuffer<K>, CollectiveError> {
    let refs: alloc::vec::Vec<&ReferenceValues> =
        inputs.iter().map(|input| &input.values).collect();
    let values = reduce_values(&refs, op)?;
    ReferenceBuffer::try_new(values, inputs[0].dtype.clone())
}

fn reduce_values(
    inputs: &[&ReferenceValues],
    op: ReduceOp,
) -> Result<ReferenceValues, CollectiveError> {
    macro_rules! collect_slices {
        ($variant:ident) => {{
            let mut slices = alloc::vec::Vec::with_capacity(inputs.len());
            for input in inputs {
                let ReferenceValues::$variant(values) = input else {
                    return Err(CollectiveError::DTypeMismatch {
                        rank: slices.len(),
                        expected: inputs[0].dtype(),
                        found: input.dtype(),
                    });
                };
                slices.push(values.as_slice());
            }
            slices
        }};
    }

    match inputs[0] {
        ReferenceValues::F32(_) => Ok(ReferenceValues::F32(reduce_float(
            &collect_slices!(F32),
            op,
            DTypeId::F32,
        )?)),
        ReferenceValues::F64(_) => Ok(ReferenceValues::F64(reduce_float(
            &collect_slices!(F64),
            op,
            DTypeId::F64,
        )?)),
        ReferenceValues::F16(_) => {
            let slices = collect_slices!(F16);
            let converted: alloc::vec::Vec<alloc::vec::Vec<f32>> = slices
                .iter()
                .map(|slice| slice.iter().map(|value| value.to_f32()).collect())
                .collect();
            let refs: alloc::vec::Vec<&[f32]> =
                converted.iter().map(alloc::vec::Vec::as_slice).collect();
            Ok(ReferenceValues::F16(
                reduce_float(&refs, op, DTypeId::F16)?
                    .into_iter()
                    .map(half::f16::from_f32)
                    .collect(),
            ))
        }
        ReferenceValues::BF16(_) => {
            let slices = collect_slices!(BF16);
            let converted: alloc::vec::Vec<alloc::vec::Vec<f32>> = slices
                .iter()
                .map(|slice| slice.iter().map(|value| value.to_f32()).collect())
                .collect();
            let refs: alloc::vec::Vec<&[f32]> =
                converted.iter().map(alloc::vec::Vec::as_slice).collect();
            Ok(ReferenceValues::BF16(
                reduce_float(&refs, op, DTypeId::BF16)?
                    .into_iter()
                    .map(half::bf16::from_f32)
                    .collect(),
            ))
        }
        ReferenceValues::U8(_) => Ok(ReferenceValues::U8(reduce_integer(
            &collect_slices!(U8),
            op,
            DTypeId::U8,
        )?)),
        ReferenceValues::U32(_) => Ok(ReferenceValues::U32(reduce_integer(
            &collect_slices!(U32),
            op,
            DTypeId::U32,
        )?)),
        ReferenceValues::I64(_) => Ok(ReferenceValues::I64(reduce_integer(
            &collect_slices!(I64),
            op,
            DTypeId::I64,
        )?)),
    }
}

trait ReferenceFloat: Copy + PartialOrd {
    fn zero() -> Self;
    fn one() -> Self;
    fn add(self, rhs: Self) -> Self;
    fn mul(self, rhs: Self) -> Self;
    fn divide(self, rhs: usize) -> Self;
}

impl ReferenceFloat for f32 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
    fn divide(self, rhs: usize) -> Self {
        self / rhs as f32
    }
}

impl ReferenceFloat for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn add(self, rhs: Self) -> Self {
        self + rhs
    }
    fn mul(self, rhs: Self) -> Self {
        self * rhs
    }
    fn divide(self, rhs: usize) -> Self {
        self / rhs as f64
    }
}

fn reduce_float<T: ReferenceFloat>(
    inputs: &[&[T]],
    op: ReduceOp,
    _dtype: DTypeId,
) -> Result<alloc::vec::Vec<T>, CollectiveError> {
    let mut output = alloc::vec::Vec::with_capacity(inputs[0].len());
    for element in 0..inputs[0].len() {
        let mut value = match op {
            ReduceOp::Sum | ReduceOp::Mean => T::zero(),
            ReduceOp::Prod => T::one(),
            ReduceOp::Max | ReduceOp::Min => inputs[0][element],
        };
        for (rank, input) in inputs.iter().enumerate() {
            if rank == 0 && matches!(op, ReduceOp::Max | ReduceOp::Min) {
                continue;
            }
            let next = input[element];
            value = match op {
                ReduceOp::Sum | ReduceOp::Mean => value.add(next),
                ReduceOp::Prod => value.mul(next),
                ReduceOp::Max => {
                    if next > value {
                        next
                    } else {
                        value
                    }
                }
                ReduceOp::Min => {
                    if next < value {
                        next
                    } else {
                        value
                    }
                }
            };
        }
        if op == ReduceOp::Mean {
            value = value.divide(inputs.len());
        }
        output.push(value);
    }
    Ok(output)
}

trait ReferenceInteger: Copy + Ord {
    fn checked_add(self, rhs: Self) -> Option<Self>;
    fn checked_mul(self, rhs: Self) -> Option<Self>;
    fn one() -> Self;
    fn zero() -> Self;
}

macro_rules! integer_impl {
    ($($type:ty),+ $(,)?) => {$(
        impl ReferenceInteger for $type {
            fn checked_add(self, rhs: Self) -> Option<Self> { self.checked_add(rhs) }
            fn checked_mul(self, rhs: Self) -> Option<Self> { self.checked_mul(rhs) }
            fn one() -> Self { 1 }
            fn zero() -> Self { 0 }
        }
    )+};
}

integer_impl!(u8, u32, i64);

fn reduce_integer<T: ReferenceInteger>(
    inputs: &[&[T]],
    op: ReduceOp,
    dtype: DTypeId,
) -> Result<alloc::vec::Vec<T>, CollectiveError> {
    if op == ReduceOp::Mean {
        return Err(CollectiveError::UnsupportedReduction { dtype, op });
    }
    let mut output = alloc::vec::Vec::with_capacity(inputs[0].len());
    for element in 0..inputs[0].len() {
        let mut value = match op {
            ReduceOp::Sum => T::zero(),
            ReduceOp::Prod => T::one(),
            ReduceOp::Max | ReduceOp::Min => inputs[0][element],
            ReduceOp::Mean => return Err(CollectiveError::UnsupportedReduction { dtype, op }),
        };
        for (rank, input) in inputs.iter().enumerate() {
            if rank == 0 && matches!(op, ReduceOp::Max | ReduceOp::Min) {
                continue;
            }
            let next = input[element];
            value = match op {
                ReduceOp::Sum => value.checked_add(next),
                ReduceOp::Prod => value.checked_mul(next),
                ReduceOp::Max => Some(core::cmp::max(value, next)),
                ReduceOp::Min => Some(core::cmp::min(value, next)),
                ReduceOp::Mean => None,
            }
            .ok_or(CollectiveError::ReductionOverflow { dtype, op, element })?;
        }
        output.push(value);
    }
    Ok(output)
}

fn concatenate<K: DType>(
    inputs: &[ReferenceBuffer<K>],
) -> Result<ReferenceValues, CollectiveError> {
    let values: alloc::vec::Vec<&ReferenceValues> =
        inputs.iter().map(|input| &input.values).collect();
    concatenate_values(&values)
}

fn concatenate_values(inputs: &[&ReferenceValues]) -> Result<ReferenceValues, CollectiveError> {
    macro_rules! concatenate {
        ($variant:ident) => {{
            let mut output = alloc::vec::Vec::new();
            for (rank, input) in inputs.iter().enumerate() {
                let ReferenceValues::$variant(values) = input else {
                    return Err(CollectiveError::DTypeMismatch {
                        rank,
                        expected: inputs[0].dtype(),
                        found: input.dtype(),
                    });
                };
                output.extend_from_slice(values);
            }
            ReferenceValues::$variant(output)
        }};
    }
    Ok(match inputs[0] {
        ReferenceValues::U8(_) => concatenate!(U8),
        ReferenceValues::U32(_) => concatenate!(U32),
        ReferenceValues::I64(_) => concatenate!(I64),
        ReferenceValues::BF16(_) => concatenate!(BF16),
        ReferenceValues::F16(_) => concatenate!(F16),
        ReferenceValues::F32(_) => concatenate!(F32),
        ReferenceValues::F64(_) => concatenate!(F64),
    })
}

fn split(
    values: ReferenceValues,
    ranks: usize,
) -> Result<alloc::vec::Vec<ReferenceValues>, CollectiveError> {
    let elements = values.len();
    if !elements.is_multiple_of(ranks) {
        return Err(CollectiveError::NonDivisible { elements, ranks });
    }
    let chunk = elements / ranks;
    macro_rules! split {
        ($values:expr, $variant:ident) => {
            $values
                .chunks(chunk)
                .map(|part| ReferenceValues::$variant(part.to_vec()))
                .collect()
        };
    }
    Ok(match values {
        ReferenceValues::U8(values) => split!(values, U8),
        ReferenceValues::U32(values) => split!(values, U32),
        ReferenceValues::I64(values) => split!(values, I64),
        ReferenceValues::BF16(values) => split!(values, BF16),
        ReferenceValues::F16(values) => split!(values, F16),
        ReferenceValues::F32(values) => split!(values, F32),
        ReferenceValues::F64(values) => split!(values, F64),
    })
}

/// Pure topology probe used by reference and planner tests.
#[derive(Debug, Clone)]
pub struct ReferenceTopology {
    devices: alloc::vec::Vec<DeviceIdentity>,
    link: LinkClass,
    layout: ProcessLayout,
}

impl ReferenceTopology {
    /// Build a topology with one uniform inter-rank link class.
    #[must_use]
    pub fn new(
        devices: alloc::vec::Vec<DeviceIdentity>,
        link: LinkClass,
        layout: ProcessLayout,
    ) -> Self {
        Self {
            devices,
            link,
            layout,
        }
    }
}

impl TopologyProbe for ReferenceTopology {
    fn identify(&self, device: DeviceId) -> Option<DeviceIdentity> {
        self.devices
            .iter()
            .find(|identity| identity.device() == device)
            .cloned()
    }

    fn link(&self, from: DeviceId, to: DeviceId) -> LinkClass {
        if from == to {
            LinkClass::SameDevice
        } else if self.identify(from).is_some() && self.identify(to).is_some() {
            self.link
        } else {
            LinkClass::Unreachable
        }
    }

    fn transport(&self) -> TransportVersion {
        TransportVersion::new("incin-reference".into(), 1, 0, 0)
    }

    fn layout(&self) -> ProcessLayout {
        self.layout.clone()
    }
}
