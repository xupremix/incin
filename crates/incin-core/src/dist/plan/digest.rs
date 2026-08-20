//! Stable hashing for group and plan identity, and the small checked-arithmetic
//! helpers the collective-plan builder shares.

use super::*;

pub(crate) fn kind_for_transition(
    transition: PlacementTransition,
    source: PlacementKind,
) -> Result<CollectiveKind, PlanError> {
    match transition {
        PlacementTransition::AllGather => Ok(CollectiveKind::AllGather),
        PlacementTransition::AllReduce | PlacementTransition::ReduceScatter => {
            let PlacementKind::Partial { reduction } = source else {
                return Err(PlanError::MissingReduction {
                    transition,
                    placement: source,
                });
            };
            if transition == PlacementTransition::AllReduce {
                Ok(CollectiveKind::AllReduce(reduction))
            } else {
                Ok(CollectiveKind::ReduceScatter(reduction))
            }
        }
        PlacementTransition::Identity | PlacementTransition::LocalShard => {
            Err(PlanError::NoCollectiveRequired { transition })
        }
    }
}

pub(crate) fn output_elements(
    kind: CollectiveKind,
    input: usize,
    ranks: usize,
) -> Result<usize, PlanError> {
    match kind {
        CollectiveKind::AllReduce(_)
        | CollectiveKind::AllToAll
        | CollectiveKind::SendRecv { .. } => {
            if matches!(kind, CollectiveKind::AllToAll) && !input.is_multiple_of(ranks) {
                return Err(CollectiveError::NonDivisible {
                    elements: input,
                    ranks,
                }
                .into());
            }
            Ok(input)
        }
        CollectiveKind::AllGather => input
            .checked_mul(ranks)
            .ok_or(ShapeError::ArithmeticOverflow {
                operation: OperationKind::Storage,
                expression: "rank-local elements * collective ranks",
            })
            .map_err(Into::into),
        CollectiveKind::ReduceScatter(_) => {
            if !input.is_multiple_of(ranks) {
                return Err(CollectiveError::NonDivisible {
                    elements: input,
                    ranks,
                }
                .into());
            }
            Ok(input / ranks)
        }
    }
}

pub(crate) fn validate_peer(
    endpoint: &'static str,
    rank: usize,
    ranks: usize,
) -> Result<(), CollectiveError> {
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

pub(crate) fn group_token(mesh: MeshId, axis: MeshAxis, members: &[usize]) -> u64 {
    let mut digest = StableDigest::new()
        .bytes(b"incin.collective.group.v1")
        .number(mesh.digest())
        .bytes(axis.name().as_bytes())
        .number(members.len() as u64);
    for &rank in members {
        digest = digest.number(rank as u64);
    }
    digest.finish()
}

pub(crate) fn plan_hash(mesh: MeshId, descriptors: &[CollectiveDescriptor]) -> u64 {
    let mut digest = StableDigest::new()
        .bytes(b"incin.collective.plan.v2")
        .number(mesh.digest())
        .number(descriptors.len() as u64);
    for descriptor in descriptors {
        digest = digest
            .number(descriptor.tag().get())
            .number(descriptor.group().token())
            .number(descriptor.group().ranks() as u64)
            .collective(descriptor.kind())
            .number(descriptor.input_elements() as u64)
            .number(descriptor.output_elements() as u64)
            .number(descriptor.input_bytes() as u64)
            .number(descriptor.output_bytes() as u64)
            .dtype(descriptor.dtype())
            .placement(descriptor.source())
            .placement(descriptor.destination())
            .number(descriptor.sequence().get())
            .number(u64::from(descriptor.stream().get()))
            .number(
                descriptor
                    .depends_on()
                    .map_or(u64::MAX, |token| token.get()),
            );
    }
    digest.finish()
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct StableDigest(u64);

impl StableDigest {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    const fn new() -> Self {
        Self(Self::OFFSET)
    }

    fn bytes(mut self, bytes: &[u8]) -> Self {
        self = self.number(bytes.len() as u64);
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(Self::PRIME);
        }
        self
    }

    fn number(self, value: u64) -> Self {
        self.bytes_raw(&value.to_le_bytes())
    }

    fn bytes_raw(mut self, bytes: &[u8]) -> Self {
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(Self::PRIME);
        }
        self
    }

    fn dtype(self, dtype: DTypeId) -> Self {
        self.bytes(dtype.name().as_bytes())
    }

    fn collective(self, kind: CollectiveKind) -> Self {
        match kind {
            CollectiveKind::AllReduce(op) => self.bytes(b"all-reduce").reduce(op),
            CollectiveKind::AllGather => self.bytes(b"all-gather"),
            CollectiveKind::ReduceScatter(op) => self.bytes(b"reduce-scatter").reduce(op),
            CollectiveKind::AllToAll => self.bytes(b"all-to-all"),
            CollectiveKind::SendRecv {
                source,
                destination,
            } => self
                .bytes(b"send-recv")
                .number(source as u64)
                .number(destination as u64),
        }
    }

    fn reduce(self, op: ReduceOp) -> Self {
        self.bytes(match op {
            ReduceOp::Sum => b"sum",
            ReduceOp::Mean => b"mean",
            ReduceOp::Max => b"max",
            ReduceOp::Min => b"min",
            ReduceOp::Prod => b"prod",
        })
    }

    fn placement(self, placement: PlacementKind) -> Self {
        match placement {
            PlacementKind::Local => self.bytes(b"local"),
            PlacementKind::Replicated => self.bytes(b"replicated"),
            PlacementKind::Sharded { axis } => self.bytes(b"sharded").number(axis as u64),
            PlacementKind::Partial { reduction } => self.bytes(b"partial").reduce(reduction),
            PlacementKind::PipelineStage { index } => {
                self.bytes(b"pipeline-stage").number(index as u64)
            }
        }
    }

    const fn finish(self) -> u64 {
        self.0
    }
}
