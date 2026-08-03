use crate::nn::StateDict;
use crate::prelude::*;
use alloc::collections::BTreeMap;
use alloc::string::{String, ToString};
use alloc::vec::Vec;
use safetensors::SafeTensors;
use std::path::Path;

/// Metadata for an individual parameter stored in a global checkpoint.
#[cfg(feature = "std")]
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct TensorCheckpointMeta {
    pub name: String,
    pub global_shape: Vec<usize>,
    pub dtype: DTypeId,
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
        dtype: DTypeId,
        placement_kind: impl Into<String>,
    ) {
        let key = name.into();
        self.tensors.insert(
            key.clone(),
            TensorCheckpointMeta {
                name: key,
                global_shape,
                dtype,
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
    dtype: DTypeId,
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

    let elem_bytes = dtype.element_size();
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

/// Loads raw storage tensors from a safetensors file into a dictionary.
pub fn load_safetensors_map<B, P>(
    path: P,
    device: &DeviceId,
) -> Result<BTreeMap<String, B::Storage<B::FloatElem>>>
where
    B: Backend,
    P: AsRef<Path>,
    B: SupportsDType<B::FloatElem>,
{
    let path_ref = path.as_ref();
    let metadata = std::fs::metadata(path_ref)?;
    let limits = ResourceLimits::model_load_defaults();
    if metadata.len() > limits.max_file_bytes {
        return Err(Error::Msg(format!(
            "Safetensors file size {} exceeds limit {} bytes",
            metadata.len(),
            limits.max_file_bytes
        )));
    }
    let buffer = std::fs::read(path_ref)
        .map_err(|e| Error::Msg(format!("Failed to read safetensors file: {}", e)))?;
    let tensors = SafeTensors::deserialize(&buffer)
        .map_err(|e| Error::Msg(format!("Safetensors deserialization failed: {:?}", e)))?;

    let mut mapped_tensors = BTreeMap::new();

    for (name, view) in tensors.tensors() {
        let shape = view.shape().to_vec();
        let bytes = view.data();
        let st_dtype = view.dtype();
        let dtype = match st_dtype {
            safetensors::Dtype::F32 => DTypeId::F32,
            safetensors::Dtype::F64 => DTypeId::F64,
            safetensors::Dtype::F16 => DTypeId::F16,
            safetensors::Dtype::BF16 => DTypeId::BF16,
            safetensors::Dtype::I64 => DTypeId::I64,
            safetensors::Dtype::U32 | safetensors::Dtype::I32 => DTypeId::U32,
            safetensors::Dtype::U8 | safetensors::Dtype::BOOL => DTypeId::U8,
            other => {
                return Err(Error::Msg(format!(
                    "Unsupported safetensors dtype {:?} for tensor {}",
                    other, name
                )));
            }
        };

        let inner = B::from_bytes::<B::FloatElem>(bytes, &shape, dtype, device)?;
        mapped_tensors.insert(name.to_string(), inner);
    }

    Ok(mapped_tensors)
}

/// Loads weights into a module from a safetensors file.
pub fn load_safetensors<B, M, P>(module: &mut M, path: P) -> Result<()>
where
    B: Backend,
    M: StateDict<B>,
    P: AsRef<Path>,
    B: SupportsDType<B::FloatElem>,
    <<B as Backend>::Device as Device>::Field: Default,
    <<B as Backend>::FloatElem as DType>::Field: Default,
{
    let map = load_safetensors_map::<B, _>(path, &DeviceId::cpu())?;
    let mut mapped_tensors = BTreeMap::new();

    for (name, storage) in map {
        let shape = B::shape(&storage);
        let tensor = Tensor::<Dyn, B>::from_parts(
            storage,
            shape,
            Default::default(),
            Default::default(),
            core::marker::PhantomData,
        )?;
        mapped_tensors.insert(name, tensor);
    }

    module.load_state_dict("", &mapped_tensors)?;
    Ok(())
}

/// Saves the module's weights to a safetensors file.
pub fn save_safetensors<B, M, P>(module: &M, path: P) -> Result<()>
where
    B: Backend,
    M: StateDict<B>,
    P: AsRef<Path>,
{
    let mut mapped_tensors = BTreeMap::new();
    module.state_dict("", &mut mapped_tensors);

    let mut raw_data: Vec<(String, Vec<usize>, safetensors::Dtype, Vec<u8>)> = Vec::new();

    for (name, tensor) in mapped_tensors {
        let bytes = B::to_bytes::<B::FloatElem>(tensor.inner())?;
        let shape = B::shape::<B::FloatElem>(tensor.inner());
        let dtype_id = B::storage_dtype::<B::FloatElem>(tensor.inner()).unwrap_or(DTypeId::F32);

        let st_dtype = match dtype_id {
            DTypeId::F32 => safetensors::Dtype::F32,
            DTypeId::F64 => safetensors::Dtype::F64,
            DTypeId::F16 => safetensors::Dtype::F16,
            DTypeId::BF16 => safetensors::Dtype::BF16,
            DTypeId::I64 => safetensors::Dtype::I64,
            DTypeId::U32 => safetensors::Dtype::U32,
            DTypeId::U8 => safetensors::Dtype::U8,
            other => {
                return Err(Error::Msg(format!(
                    "Unsupported DTypeId {:?} for safetensors export",
                    other
                )));
            }
        };

        raw_data.push((name, shape, st_dtype, bytes));
    }

    let mut data_map: BTreeMap<String, safetensors::tensor::TensorView> = BTreeMap::new();
    for (name, shape, st_dtype, bytes) in &raw_data {
        let view = safetensors::tensor::TensorView::new(*st_dtype, shape.clone(), bytes)
            .map_err(|e| Error::Msg(format!("TensorView creation failed: {:?}", e)))?;
        data_map.insert(name.clone(), view);
    }

    let serialized = safetensors::serialize(&data_map, &None)
        .map_err(|e| Error::Msg(format!("Safetensors serialization failed: {:?}", e)))?;

    let path_ref = path.as_ref();
    let tmp_path = path_ref.with_extension(format!(
        "{}.tmp",
        path_ref
            .extension()
            .and_then(|s| s.to_str())
            .unwrap_or("safetensors")
    ));

    std::fs::write(&tmp_path, serialized)
        .map_err(|e| Error::Msg(format!("Failed to write safetensors tmp file: {}", e)))?;

    std::fs::rename(&tmp_path, path_ref)
        .map_err(|e| Error::Msg(format!("Failed to finalize safetensors rename: {}", e)))?;

    Ok(())
}

/// Saves a full model checkpoint including global manifest and weights file.
#[cfg(feature = "std")]
pub fn save_checkpoint<B, M, P>(module: &M, dir_path: P, world_size: usize) -> Result<()>
where
    B: Backend,
    M: StateDict<B>,
    P: AsRef<Path>,
{
    let dir = dir_path.as_ref();
    std::fs::create_dir_all(dir)?;

    let mut mapped_tensors = BTreeMap::new();
    module.state_dict("", &mut mapped_tensors);

    let mut manifest = GlobalCheckpointManifest::new(world_size);
    for (name, tensor) in &mapped_tensors {
        let shape = B::shape::<B::FloatElem>(tensor.inner());
        let dtype_id = B::storage_dtype::<B::FloatElem>(tensor.inner()).unwrap_or(DTypeId::F32);
        manifest.add_tensor(name.clone(), shape, dtype_id, "Local");
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
    B: Backend,
    M: StateDict<B>,
    P: AsRef<Path>,
    B: SupportsDType<B::FloatElem>,
    <<B as Backend>::Device as Device>::Field: Default,
    <<B as Backend>::FloatElem as DType>::Field: Default,
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
    let raw_bytes = std::fs::read(&weights_path)
        .map_err(|e| Error::Msg(format!("Failed to read safetensors weights: {}", e)))?;
    let st = safetensors::SafeTensors::deserialize(&raw_bytes)
        .map_err(|e| Error::Msg(format!("Safetensors deserialization failed: {:?}", e)))?;

    let mut current_dict = BTreeMap::new();
    module.state_dict("", &mut current_dict);

    let mut resharded_tensors = BTreeMap::new();

    for (name, current_param) in current_dict {
        let st_view = st.tensor(&name).map_err(|e| {
            Error::Msg(format!(
                "Parameter {} missing from checkpoint safetensors: {:?}",
                name, e
            ))
        })?;

        let meta = manifest
            .tensors
            .get(&name)
            .ok_or_else(|| Error::Msg(format!("Parameter {} missing from manifest", name)))?;

        let global_shape = st_view.shape().to_vec();
        let target_shape = B::shape::<B::FloatElem>(current_param.inner());
        let bytes = st_view.data();
        let dtype_id = meta.dtype;

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
                dtype_id,
                shard_axis,
                rank,
                target_world_size,
            )?
        };

        if final_shape != target_shape {
            return Err(Error::Msg(format!(
                "Resharded shape {:?} does not match target shape {:?} for parameter {}",
                final_shape, target_shape, name
            )));
        }

        let storage =
            B::from_bytes::<B::FloatElem>(&final_bytes, &final_shape, dtype_id, &DeviceId::cpu())?;
        let tensor = Tensor::<Dyn, B>::from_parts(
            storage,
            final_shape,
            Default::default(),
            Default::default(),
            core::marker::PhantomData,
        )?;
        resharded_tensors.insert(name, tensor);
    }

    module.load_state_dict("", &resharded_tensors)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    // Integration tests for checkpoint save, load, and resharding are located in `crates/incin-core/tests/checkpoint_reshard.rs`.
}
