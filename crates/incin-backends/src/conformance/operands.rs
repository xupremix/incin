//! Building the operands a fixture poses a tuple with.
//!
//! Split out of `fixtures.rs`, which holds the family tables that say *which*
//! operands an operation takes. This module answers the other half: what those
//! operands are made of, for a given rank, layout, dtype and role.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use half::{bf16, f16};

use incin_core::tensor::dtype::{DTypeDescriptor, DTypeId};

use crate::conformance::AdvertisedTuple;
use crate::conformance::fixtures::{Operands, Role};
use crate::cpu::{CpuBuffer, CpuStorage};

/// Extents for a rank, small and unequal.
///
/// Unequal on purpose: a kernel that transposes when it should not, or that
/// reads an extent off the wrong axis, produces the right answer on a square
/// operand and the wrong one here.
fn extents(rank: usize) -> Vec<usize> {
    const LADDER: [usize; 4] = [2, 3, 2, 2];
    LADDER[..rank.min(LADDER.len())].to_vec()
}

/// The extents an operand actually carries once its layout has been applied.
///
/// The strided operand is a transpose of the contiguous one, so its first two
/// extents are swapped. Anything naming a target shape has to ask for the shape
/// the operand really has, or a reshape to its own extents stops being an
/// identity halfway through the layout axis.
pub(crate) fn materialized_extents(tuple: &AdvertisedTuple) -> Vec<usize> {
    let mut dims = extents(tuple.rank);
    if tuple.layout != incin_core::exec::LayoutClass::Contiguous && tuple.rank >= 2 {
        dims.swap(0, 1);
    }
    dims
}

/// A buffer of `length` values in `dtype`.
///
/// Values are positive and away from zero so that a domain-restricted operation
/// (`log`, `sqrt`, `acosh`, a division) has a defined answer, and they are whole
/// numbers so that a float operand converts to an integer one without the
/// narrowing that `int_to_vec` refuses on a fractional value. A tuple that
/// fails here is a harness gap, never a backend finding, so it is reported as
/// [`Coverage::Unbuildable`].
fn buffer(dtype: DTypeDescriptor, length: usize) -> Result<CpuBuffer, String> {
    let magnitudes: Vec<f64> = (0..length).map(|index| 1.0 + index as f64).collect();

    Ok(match dtype.builtin_id() {
        Some(DTypeId::F32) => {
            CpuBuffer::F32(magnitudes.iter().map(|value| *value as f32).collect())
        }
        Some(DTypeId::F64) => CpuBuffer::F64(magnitudes),
        Some(DTypeId::F16) => CpuBuffer::F16(
            magnitudes
                .iter()
                .map(|value| f16::from_f64(*value))
                .collect(),
        ),
        Some(DTypeId::BF16) => CpuBuffer::BF16(
            magnitudes
                .iter()
                .map(|value| bf16::from_f64(*value))
                .collect(),
        ),
        Some(DTypeId::U8) => {
            CpuBuffer::U8((0..length).map(|index| (index % 7 + 1) as u8).collect())
        }
        Some(DTypeId::U32) => {
            CpuBuffer::U32((0..length).map(|index| (index % 7 + 1) as u32).collect())
        }
        Some(DTypeId::I64) => {
            CpuBuffer::I64((0..length).map(|index| (index % 7 + 1) as i64).collect())
        }
        Some(DTypeId::Bool) => {
            CpuBuffer::Bool((0..length).map(|index| (index % 2) as u8).collect())
        }
        other => {
            return Err(alloc::format!(
                "no operand builder for {other:?}; block-encoded dtypes need their own"
            ));
        }
    })
}

/// The buffer one operand carries, given its role.
///
/// Index roles are filled with zeros rather than with the ladder. Zero is the
/// only value guaranteed in range for every table extent the harness builds,
/// and an out-of-range index would be reported as the backend refusing a row it
/// advertises when the fault was the fixture's.
fn role_buffer(tuple: &AdvertisedTuple, role: Role, length: usize) -> Result<CpuBuffer, String> {
    match role {
        Role::Tuple => buffer(tuple.dtype, length),
        Role::Mask => Ok(CpuBuffer::Bool(alloc::vec![1; length])),
        Role::Float | Role::FloatMatrix => buffer(DTypeId::F32.descriptor(), length),
        Role::Index | Role::IndexVector => Ok(CpuBuffer::I64(alloc::vec![0; length])),
        // The shaped roles fix an extent, never a dtype. A convolution weight
        // and its activation must agree on dtype for the row to mean anything,
        // and so must a norm's parameters, so all of them read the tuple's.
        Role::Paired { .. }
        | Role::ConvWeight { .. }
        | Role::ConvTransposeWeight { .. }
        | Role::OutputVector
        | Role::ChannelVector
        | Role::TrailingVector
        | Role::LinearWeight => buffer(tuple.dtype, length),
    }
}

/// Output channels for every convolution fixture.
///
/// Five, which is on no rank ladder and equal to no extent the harness builds.
/// A weight whose two channel axes were swapped still satisfies the inference
/// when the input's channel extent happens to equal the output count, and the
/// strided rank-four activation carries a channel extent of two. Picking a
/// number outside the ladder is the same discipline `extents` documents.
const OUT_CHANNELS: usize = 5;

/// A convolution filter bank for an activation of these extents.
///
/// `[out, in, ..ones]` for a forward convolution and `[in, out, ..ones]` for a
/// transposed one, which is the only difference between them. The channel
/// extent is read at `len - 1 - spatial`, the axis `inference.rs` reads it at,
/// and the kernel is unit in every spatial axis so the output extent equals
/// the input's for any extent the ladder produces.
///
/// Groups stay at one throughout. The forward inference reads the bias against
/// `weight[0]` and the transposed one against `weight[1] * groups`, so a
/// single bias role serves both only while that factor is one.
fn conv_weight(input: &[usize], spatial: usize, transposed: bool) -> Result<Vec<usize>, String> {
    let Some(channel_axis) = input.len().checked_sub(spatial + 1) else {
        return Err(alloc::format!(
            "a convolution over {spatial} spatial axes needs an activation of at \
             least rank {}, and this row advertises rank {}",
            spatial + 1,
            input.len()
        ));
    };

    let channels = input[channel_axis];
    let (leading, trailing) = if transposed {
        (channels, OUT_CHANNELS)
    } else {
        (OUT_CHANNELS, channels)
    };

    let mut dims = alloc::vec![1; spatial + 2];
    dims[0] = leading;
    dims[1] = trailing;
    Ok(dims)
}

/// The tuple's batch extents followed by two named ones.
///
/// The batch prefix is the ladder truncated to whatever the rank leaves after
/// the two trailing axes, so every `Paired` operand of one tuple shares it and
/// a batched product's leading extents agree by construction. Below rank two
/// there is no room for the pair at all, which for `addmm` is exactly what its
/// declared floor of one is for: `rules.rs` records that the floor exists so a
/// per-column rank-one *addend* is a legal operand of a rank-two operation, and
/// there is no rank-one invocation to pose.
fn paired(rank: usize, rows: usize, columns: usize) -> Result<Vec<usize>, String> {
    if rank < 2 {
        return Err(alloc::format!(
            "an operand of two named extents needs two axes and this row \
             advertises rank {rank}; a floor below two speaks for a broadcast \
             addend or a bias, not for an invocation that can be posed"
        ));
    }
    let mut dims = extents(rank)[..rank - 2].to_vec();
    dims.extend_from_slice(&[rows, columns]);
    Ok(dims)
}

/// A `[out, in]` projection against an input whose final extent is `in`.
fn linear_weight(input: &[usize]) -> Result<Vec<usize>, String> {
    input
        .last()
        .map(|width| alloc::vec![OUT_CHANNELS, *width])
        .ok_or_else(|| {
            "a projection reads the input's final extent and a rank-zero operand has none"
                .to_string()
        })
}

/// A per-channel vector for an activation of these extents.
///
/// Axis one, because `BatchNormAttributes::validate` reads `input[1]` outright
/// rather than counting back from the end. Below rank two there is no such
/// axis, and the row's floor of one speaks for these vectors themselves rather
/// than for an activation that could carry them.
fn channel_vector(input: &[usize]) -> Result<Vec<usize>, String> {
    input
        .get(1)
        .map(|channels| alloc::vec![*channels])
        .ok_or_else(|| {
            alloc::format!(
                "a per-channel vector is read at axis one and rank {} has none; \
                 the row's floor speaks for the vector, not for the activation",
                input.len()
            )
        })
}

/// The input's final extent as a one-axis vector.
///
/// An RMS norm weight must match the last input extent, and a layer norm over
/// a one-axis normalized shape asks for the same thing.
fn trailing_vector(input: &[usize]) -> Result<Vec<usize>, String> {
    input
        .last()
        .map(|extent| alloc::vec![*extent])
        .ok_or_else(|| {
            "a normalized suffix needs a final extent and a rank-zero operand has none".to_string()
        })
}

/// One operand matching `tuple`'s dtype, rank and layout.
///
/// The strided operand is produced by transposing a contiguous one rather than
/// by constructing strides directly. Transposing is the only way to get a
/// non-contiguous `CpuStorage` through the public surface, and it has the
/// property the layout claim is actually about: the buffer is untouched and the
/// metadata no longer describes a row-major walk.
pub(crate) fn operand(
    tuple: &AdvertisedTuple,
    operands: Operands,
    role: Role,
) -> Result<CpuStorage, String> {
    use incin_core::exec::LayoutClass;

    if operands == Operands::UnaryAxis && tuple.rank == 0 {
        return Err(alloc::format!(
            "{} names an axis and a rank-0 operand has none",
            tuple.operation
        ));
    }

    // The shaped roles are sized against the extents the activation really
    // carries, which for a strided tuple is the transpose of the ladder. Sizing
    // them against the ladder instead would build a weight the inference
    // rejects, and the report would read that as a backend finding.
    let activation = materialized_extents(tuple);
    let dims = match (operands, role) {
        (_, Role::IndexVector) => alloc::vec![2],
        (_, Role::FloatMatrix) => alloc::vec![2, 3],
        (_, Role::ConvWeight { spatial }) => conv_weight(&activation, spatial, false)?,
        (_, Role::ConvTransposeWeight { spatial }) => conv_weight(&activation, spatial, true)?,
        (_, Role::OutputVector) => alloc::vec![OUT_CHANNELS],
        // Built with its trailing pair reversed when the tuple is strided, so
        // that transposing it back below leaves the operand carrying the shape
        // the product needs while its metadata no longer describes a row-major
        // walk. Transposing the first two axes instead, as every other operand
        // here does, would permute the batch extents of one side and not the
        // other.
        (_, Role::Paired { rows, columns }) if tuple.layout != LayoutClass::Contiguous => {
            paired(tuple.rank, columns, rows)?
        }
        (_, Role::Paired { rows, columns }) => paired(tuple.rank, rows, columns)?,
        (_, Role::LinearWeight) => linear_weight(&activation)?,
        (_, Role::ChannelVector) => channel_vector(&activation)?,
        (_, Role::TrailingVector) => trailing_vector(&activation)?,
        (Operands::UnaryScalar, _) => alloc::vec![1; tuple.rank],
        _ => extents(tuple.rank),
    };
    let length: usize = dims.iter().product::<usize>().max(1);
    let data = role_buffer(tuple, role, length)?;
    let contiguous = CpuStorage::try_from_contiguous(data, &dims)
        .map_err(|error| alloc::format!("could not build a contiguous operand: {error}"))?;

    match tuple.layout {
        LayoutClass::Contiguous => Ok(contiguous),
        // `paired` refused anything below rank two, so the two axes exist.
        _ if matches!(role, Role::Paired { .. }) => contiguous
            .transpose(tuple.rank - 2, tuple.rank - 1)
            .map_err(|error| alloc::format!("could not build a strided pair: {error}")),
        // An operand whose shape the role fixed is not the one the layout claim
        // is about; the tuple's layout describes the operand that carries its
        // dtype, and transposing an index vector says nothing.
        _ if !role.follows_tuple_shape() => Ok(contiguous),
        _ if tuple.rank < 2 => Err(alloc::format!(
            "rank {} has no two axes to transpose, so no strided operand exists \
             through the public surface",
            tuple.rank
        )),
        _ => contiguous
            .transpose(0, 1)
            .map_err(|error| alloc::format!("could not build a strided operand: {error}")),
    }
}
