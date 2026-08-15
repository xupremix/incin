use crate::backend_authoring::{Execute, op};
use crate::err::ErrorMessage;
use crate::err::{Error, Result};
use crate::io::ResourceLimits;
use crate::nn::{VisitState, VisitStateMut};
use crate::shapes::ShapeBuf;
use crate::tensor::dtype::{DTypeDescriptor, DTypeId, DTypeKey, DTypeKind, StorageEncoding};
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use std::path::Path;

/// Metadata for an individual parameter stored in a global checkpoint.
#[cfg(feature = "std")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct CheckpointDType {
    /// Stable logical identity of the dtype.
    pub key: DTypeKey,
    /// Semantic category, retained independently from the logical name.
    pub kind: DTypeKind,
    /// Physical storage/block encoding used by the checkpoint bytes.
    pub encoding: StorageEncoding,
}

#[cfg(feature = "std")]
impl CheckpointDType {
    pub fn from_descriptor(dtype: DTypeDescriptor) -> Self {
        Self {
            key: dtype.key(),
            kind: dtype.kind(),
            encoding: dtype.encoding(),
        }
    }

    /// Resolve a manifest dtype into the runtime descriptor after validating
    /// all semantic and physical fields against the built-in registry.
    pub fn descriptor(&self) -> Result<DTypeDescriptor> {
        let descriptor = match self.key.name() {
            "u8" => DTypeId::U8.descriptor(),
            "u32" => DTypeId::U32.descriptor(),
            "i64" => DTypeId::I64.descriptor(),
            "bf16" => DTypeId::BF16.descriptor(),
            "f16" => DTypeId::F16.descriptor(),
            "f32" => DTypeId::F32.descriptor(),
            "f64" => DTypeId::F64.descriptor(),
            "q8_0" => DTypeId::Q8_0.descriptor(),
            "bool" => DTypeId::Bool.descriptor(),
            _ => {
                return Err(Error::Msg(format!(
                    "Unsupported checkpoint dtype key {}:{}:{}",
                    self.key.namespace(),
                    self.key.name(),
                    self.key.version()
                )));
            }
        };
        if descriptor.key() != self.key
            || descriptor.kind() != self.kind
            || descriptor.encoding() != self.encoding
        {
            return Err(Error::Msg(format!(
                "Checkpoint dtype metadata does not match registered dtype {}",
                self.key.name()
            )));
        }
        Ok(descriptor)
    }
}

#[cfg(feature = "std")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct TensorCheckpointMeta {
    pub name: String,
    pub global_shape: Vec<usize>,
    /// Explicit semantic and physical dtype record; schema version is independent.
    pub dtype: CheckpointDType,
    pub placement_kind: String,
}

/// Global checkpoint manifest recording overall topology and global parameter shapes.
#[cfg(feature = "std")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct GlobalCheckpointManifest {
    pub version: u32,
    pub world_size: usize,
    pub tensors: BTreeMap<String, TensorCheckpointMeta>,
}

#[cfg(feature = "std")]
impl GlobalCheckpointManifest {
    /// Create a new global checkpoint manifest for a specified world size.
    pub fn new(world_size: usize) -> Self {
        Self {
            version: 1,
            world_size,
            tensors: BTreeMap::new(),
        }
    }

    /// Record parameter metadata in the manifest.
    pub fn add_tensor(
        &mut self,
        name: impl Into<String>,
        global_shape: Vec<usize>,
        dtype: impl Into<DTypeDescriptor>,
        placement_kind: impl Into<String>,
    ) {
        let key = name.into();
        self.tensors.insert(
            key.clone(),
            TensorCheckpointMeta {
                name: key,
                global_shape,
                dtype: CheckpointDType::from_descriptor(dtype.into()),
                placement_kind: placement_kind.into(),
            },
        );
    }
}

/// Saves a global checkpoint manifest to a JSON file.
#[cfg(feature = "std")]
pub fn save_checkpoint_manifest<P: AsRef<Path>>(
    manifest: &GlobalCheckpointManifest,
    path: P,
) -> Result<()> {
    let json = serde_json::to_string_pretty(manifest)
        .map_err(|e| Error::Msg(format!("Failed to serialize checkpoint manifest: {}", e)))?;
    std::fs::write(path.as_ref(), json)
        .map_err(|e| Error::Msg(format!("Failed to write checkpoint manifest: {}", e)))?;
    Ok(())
}

/// Loads and validates a global checkpoint manifest from a JSON file.
#[cfg(feature = "std")]
pub fn load_checkpoint_manifest<P: AsRef<Path>>(path: P) -> Result<GlobalCheckpointManifest> {
    let path_ref = path.as_ref();
    let metadata = std::fs::metadata(path_ref)?;
    let limits = ResourceLimits::model_load_defaults();
    if metadata.len() > limits.max_header_bytes {
        return Err(Error::Msg(format!(
            "Checkpoint manifest size {} exceeds header limit {} bytes",
            metadata.len(),
            limits.max_header_bytes
        )));
    }
    let data = std::fs::read_to_string(path_ref)
        .map_err(|e| Error::Msg(format!("Failed to read checkpoint manifest: {}", e)))?;
    let manifest: GlobalCheckpointManifest = serde_json::from_str(&data)
        .map_err(|e| Error::Msg(format!("Failed to parse checkpoint manifest: {}", e)))?;
    Ok(manifest)
}

/// Slices a contiguous multidimensional byte array along a target sharded axis for a given rank.
#[cfg(feature = "std")]
pub fn slice_bytes_for_rank(
    bytes: &[u8],
    global_shape: &[usize],
    dtype: DTypeDescriptor,
    shard_axis: usize,
    rank: usize,
    world_size: usize,
) -> Result<(Vec<u8>, Vec<usize>)> {
    if shard_axis >= global_shape.len() {
        return Err(Error::Msg(format!(
            "Shard axis {} out of bounds for shape {:?}",
            shard_axis, global_shape
        )));
    }
    let global_dim = global_shape[shard_axis];
    if world_size == 0 || !global_dim.is_multiple_of(world_size) {
        return Err(Error::Msg(format!(
            "Global dimension {} along axis {} not divisible by world size {}",
            global_dim, shard_axis, world_size
        )));
    }
    if rank >= world_size {
        return Err(Error::Msg(format!(
            "Rank {} out of bounds for world size {}",
            rank, world_size
        )));
    }

    let local_shard_dim = global_dim / world_size;
    let mut local_shape = global_shape.to_vec();
    local_shape[shard_axis] = local_shard_dim;

    let elem_bytes = dtype
        .encoding()
        .scalar_bytes()
        .ok_or_else(|| Error::UnsupportedDType {
            dtype,
            backend: "safetensors",
            op: "shard",
        })?;
    let outer_stride = crate::shapes::ShapeBuf::from_slice(&global_shape[..shard_axis])
        .checked_numel(crate::shapes::error::OperationKind::Storage)?;
    let inner_stride = crate::shapes::ShapeBuf::from_slice(&global_shape[shard_axis + 1..])
        .checked_numel(crate::shapes::error::OperationKind::Storage)?;

    let checked_mul = |lhs: usize, rhs: usize, expression: &'static str| {
        lhs.checked_mul(rhs)
            .ok_or(crate::shapes::error::ShapeError::ArithmeticOverflow {
                operation: crate::shapes::error::OperationKind::Storage,
                expression,
            })
    };
    let elem_per_shard_block = checked_mul(
        local_shard_dim,
        inner_stride,
        "local shard dimension * inner stride",
    )?;
    let bytes_per_shard_block = checked_mul(
        elem_per_shard_block,
        elem_bytes,
        "shard elements * element byte width",
    )?;
    let elem_per_global_block = checked_mul(
        global_dim,
        inner_stride,
        "global shard dimension * inner stride",
    )?;
    let bytes_per_global_block = checked_mul(
        elem_per_global_block,
        elem_bytes,
        "global block elements * element byte width",
    )?;

    let output_bytes = checked_mul(
        outer_stride,
        bytes_per_shard_block,
        "outer stride * shard block bytes",
    )?;
    let mut sliced_bytes = Vec::with_capacity(output_bytes);

    for o in 0..outer_stride {
        let global_offset = checked_mul(
            o,
            bytes_per_global_block,
            "outer index * global block bytes",
        )?
        .checked_add(checked_mul(
            rank,
            bytes_per_shard_block,
            "rank * shard block bytes",
        )?)
        .ok_or(crate::shapes::error::ShapeError::ArithmeticOverflow {
            operation: crate::shapes::error::OperationKind::Storage,
            expression: "global block offset + rank block offset",
        })?;
        let end_offset = global_offset.checked_add(bytes_per_shard_block).ok_or(
            crate::shapes::error::ShapeError::ArithmeticOverflow {
                operation: crate::shapes::error::OperationKind::Storage,
                expression: "shard byte offset + shard byte length",
            },
        )?;
        if end_offset > bytes.len() {
            return Err(Error::Msg(format!(
                "Byte slicing offset {}..{} out of buffer len {}",
                global_offset,
                end_offset,
                bytes.len()
            )));
        }
        sliced_bytes.extend_from_slice(&bytes[global_offset..end_offset]);
    }

    Ok((sliced_bytes, local_shape))
}

/// Parses a safetensors file into backend-neutral owned state.
pub fn load_safetensors_snapshot<P: AsRef<Path>>(path: P) -> Result<crate::nn::StateSnapshot> {
    crate::serialize::deserialize_snapshot_safetensors(path.as_ref())
        .map_err(|e| Error::Msg(format!("Safetensors deserialization failed: {}", e)))
}

/// Loads weights into a module from a safetensors file.
pub fn load_safetensors<B, M, P>(module: &mut M, path: P) -> Result<()>
where
    B: crate::tensor::backend::VariableBackend,
    M: VisitState<B> + VisitStateMut<B>,
    P: AsRef<Path>,
{
    let snapshot = load_safetensors_snapshot(path)?;
    crate::nn::load_state::<B, _>(module, &snapshot)
}

/// Saves the module's weights to a safetensors file.
pub fn save_safetensors<B, M, P>(module: &M, path: P) -> Result<()>
where
    B: crate::tensor::backend::VariableBackend,
    M: VisitState<B>,
    P: AsRef<Path>,
{
    let snapshot = crate::nn::collect_state::<B, _>(module)?;
    crate::serialize::serialize_snapshot_safetensors(&snapshot, path.as_ref())
        .map_err(|e| Error::Msg(format!("Safetensors serialization failed: {}", e)))
}

/// Saves a full model checkpoint including global manifest and weights file.
#[cfg(feature = "std")]
pub fn save_checkpoint<B, M, P>(module: &M, dir_path: P, world_size: usize) -> Result<()>
where
    B: crate::tensor::backend::VariableBackend,
    M: VisitState<B>,
    P: AsRef<Path>,
{
    let dir = dir_path.as_ref();
    std::fs::create_dir_all(dir)?;

    let snapshot = crate::nn::collect_state::<B, _>(module)?;

    let mut manifest = GlobalCheckpointManifest::new(world_size);
    for (name, value) in snapshot.iter() {
        manifest.add_tensor(
            name.to_string(),
            value.shape().dims().to_vec(),
            value.dtype(),
            "Local",
        );
    }

    let manifest_path = dir.join("manifest.json");
    save_checkpoint_manifest(&manifest, manifest_path)?;

    let weights_path = dir.join("model.safetensors");
    save_safetensors::<B, M, _>(module, weights_path)?;

    Ok(())
}

/// Loads a checkpoint with explicit cross-mesh resharding for the target rank and topology.
#[cfg(feature = "std")]
pub fn load_resharded_checkpoint<B, M, P>(
    module: &mut M,
    dir_path: P,
    rank: usize,
    target_world_size: usize,
) -> Result<()>
where
    B: crate::tensor::backend::VariableBackend + Execute<op::TensorFromBytes>,
    M: VisitState<B> + VisitStateMut<B>,
    P: AsRef<Path>,
{
    let dir = dir_path.as_ref();
    let manifest_path = dir.join("manifest.json");
    let manifest = load_checkpoint_manifest(&manifest_path)?;

    if rank >= target_world_size {
        return Err(Error::Msg(format!(
            "Target rank {} out of bounds for target world size {}",
            rank, target_world_size
        )));
    }

    let weights_path = dir.join("model.safetensors");
    let checkpoint = crate::serialize::deserialize_snapshot_safetensors(&weights_path)
        .map_err(|e| Error::Msg(format!("Safetensors deserialization failed: {}", e)))?;
    let current = crate::nn::collect_state::<B, _>(module)?;
    let expected_paths: BTreeSet<_> = current.iter().map(|(path, _)| path).collect();
    let checkpoint_paths: BTreeSet<_> = checkpoint.iter().map(|(path, _)| path).collect();
    if expected_paths != checkpoint_paths {
        let missing = expected_paths
            .difference(&checkpoint_paths)
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        let unexpected = checkpoint_paths
            .difference(&expected_paths)
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        return Err(Error::InvalidModuleState {
            operation: "load checkpoint",
            reason: ErrorMessage::new(format!(
                "checkpoint paths differ: missing {:?}, unexpected {:?}",
                missing, unexpected
            )),
        });
    }
    let manifest_paths: BTreeSet<_> = manifest.tensors.keys().map(String::as_str).collect();
    let manifest_expected: BTreeSet<_> = expected_paths.iter().map(|path| path.as_str()).collect();
    if manifest_paths != manifest_expected {
        return Err(Error::InvalidModuleState {
            operation: "load checkpoint",
            reason: ErrorMessage::new("checkpoint manifest paths differ from model state"),
        });
    }
    let mut resharded = crate::nn::StateSnapshot::new();

    for (name, current_value) in current.iter() {
        let st_value = checkpoint.get(name).ok_or_else(|| {
            Error::Msg(format!(
                "Parameter {} missing from checkpoint safetensors: {:?}",
                name, "missing"
            ))
        })?;

        let meta = manifest
            .tensors
            .get(name.as_str())
            .ok_or_else(|| Error::Msg(format!("Parameter {} missing from manifest", name)))?;

        let global_shape = st_value.shape().dims().to_vec();
        let target_shape = current_value.shape().dims();
        let bytes = st_value.bytes();
        let dtype_desc = meta.dtype.descriptor()?;
        if global_shape != meta.global_shape {
            return Err(Error::InvalidModuleState {
                operation: "load checkpoint",
                reason: ErrorMessage::new(format!(
                    "manifest shape {:?} does not match checkpoint shape {:?} for {}",
                    meta.global_shape, global_shape, name
                )),
            });
        }
        if st_value.dtype() != dtype_desc {
            return Err(Error::InvalidModuleState {
                operation: "load checkpoint",
                reason: ErrorMessage::new(format!(
                    "manifest dtype {} does not match checkpoint dtype {} for {}",
                    dtype_desc.name(),
                    st_value.dtype().name(),
                    name
                )),
            });
        }

        let (final_bytes, final_shape) = if global_shape == target_shape {
            (bytes.to_vec(), global_shape)
        } else {
            let mut diff_axes = Vec::new();
            for (idx, (&g_dim, &t_dim)) in global_shape.iter().zip(target_shape.iter()).enumerate()
            {
                if g_dim != t_dim {
                    diff_axes.push(idx);
                }
            }
            if diff_axes.len() != 1 || global_shape.len() != target_shape.len() {
                return Err(Error::Msg(format!(
                    "Incompatible global shape {:?} and target local shape {:?} for parameter {}",
                    global_shape, target_shape, name
                )));
            }
            let shard_axis = diff_axes[0];
            let g_dim = global_shape[shard_axis];
            let t_dim = target_shape[shard_axis];

            if g_dim % t_dim != 0 || g_dim / t_dim != target_world_size {
                return Err(Error::Msg(format!(
                    "Global dim {} on axis {} incompatible with target dim {} and world size {} for parameter {}",
                    g_dim, shard_axis, t_dim, target_world_size, name
                )));
            }

            slice_bytes_for_rank(
                bytes,
                &global_shape,
                dtype_desc,
                shard_axis,
                rank,
                target_world_size,
            )?
        };

        if final_shape.as_slice() != target_shape {
            return Err(Error::Msg(format!(
                "Resharded shape {:?} does not match target shape {:?} for parameter {}",
                final_shape, target_shape, name
            )));
        }

        let path = crate::nn::StatePath::new(name.as_str())
            .map_err(|e| Error::Msg(format!("Invalid checkpoint path {}: {}", name, e)))?;
        let value = crate::nn::StateValue::new(
            ShapeBuf::from_slice(&final_shape),
            dtype_desc,
            final_bytes,
            st_value.role(),
        )?;
        resharded.insert(path, value)?;
    }

    crate::nn::load_state::<B, _>(module, &resharded)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    // Integration tests for checkpoint save, load, and resharding are located in `crates/incin-core/tests/checkpoint_reshard.rs`.
}
