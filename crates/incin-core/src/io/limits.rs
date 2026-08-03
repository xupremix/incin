//! Resource limits and bounded I/O checks for model and data parsers (`SEC-007`).

use crate::prelude::{Error, Result};

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// Framework-wide resource constraints applied to untrusted headers, data formats,
/// tensor dimensions, and model files.
pub struct ResourceLimits {
    /// Maximum allowed total file bytes.
    pub max_file_bytes: u64,
    /// Maximum allowed metadata/header bytes before tensor payload.
    pub max_header_bytes: u64,
    /// Maximum number of tensors in a file/archive.
    pub max_tensor_count: usize,
    /// Maximum number of metadata entries.
    pub max_metadata_entries: usize,
    /// Maximum length in bytes for a name string.
    pub max_name_bytes: usize,
    /// Maximum rank (number of dimensions) of a single tensor.
    pub max_rank: usize,
    /// Maximum value for a single dimension length.
    pub max_dimension: u64,
    /// Maximum bytes for a single tensor payload.
    pub max_tensor_bytes: u64,
    /// Maximum total bytes across all tensors in a container.
    pub max_total_tensor_bytes: u64,
    /// Maximum graph nodes in an imported format.
    pub max_graph_nodes: usize,
    /// Maximum graph edges in an imported format.
    pub max_graph_edges: usize,
    /// Maximum nesting depth for structures or recursive types.
    pub max_nesting_depth: usize,
    /// Maximum byte length of a general text string.
    pub max_string_bytes: usize,
    /// Maximum archive entries.
    pub max_archive_entries: usize,
    /// Maximum total expanded archive bytes.
    pub max_archive_expanded_bytes: u64,
}

impl Default for ResourceLimits {
    fn default() -> Self {
        Self::model_load_defaults()
    }
}

impl ResourceLimits {
    /// Strict limits for fast inspection/metadata extraction without loading full payloads.
    pub fn inspection_defaults() -> Self {
        Self {
            max_file_bytes: 100 * 1024 * 1024 * 1024,
            max_header_bytes: 64 * 1024 * 1024,
            max_tensor_count: 50_000,
            max_metadata_entries: 50_000,
            max_name_bytes: 4096,
            max_rank: 32,
            max_dimension: 1_000_000_000,
            max_tensor_bytes: 64 * 1024 * 1024 * 1024,
            max_total_tensor_bytes: 100 * 1024 * 1024 * 1024,
            max_graph_nodes: 100_000,
            max_graph_edges: 500_000,
            max_nesting_depth: 64,
            max_string_bytes: 1024 * 1024,
            max_archive_entries: 100_000,
            max_archive_expanded_bytes: 100 * 1024 * 1024 * 1024,
        }
    }

    /// Default limits for loading models into memory.
    pub fn model_load_defaults() -> Self {
        Self::inspection_defaults()
    }

    /// Strict limits for compile-time macro expansion or metadata evaluation.
    pub fn compile_time_defaults() -> Self {
        Self {
            max_file_bytes: 512 * 1024 * 1024,
            max_header_bytes: 16 * 1024 * 1024,
            max_tensor_count: 5_000,
            max_metadata_entries: 5_000,
            max_name_bytes: 2048,
            max_rank: 16,
            max_dimension: 100_000_000,
            max_tensor_bytes: 256 * 1024 * 1024,
            max_total_tensor_bytes: 512 * 1024 * 1024,
            max_graph_nodes: 10_000,
            max_graph_edges: 50_000,
            max_nesting_depth: 32,
            max_string_bytes: 64 * 1024,
            max_archive_entries: 10_000,
            max_archive_expanded_bytes: 512 * 1024 * 1024,
        }
    }

    /// High-capacity limits for trusted local multi-gigabyte models.
    pub fn trusted_local_large_model() -> Self {
        Self {
            max_file_bytes: 1_000 * 1024 * 1024 * 1024,
            max_header_bytes: 512 * 1024 * 1024,
            max_tensor_count: 500_000,
            max_metadata_entries: 500_000,
            max_name_bytes: 8192,
            max_rank: 64,
            max_dimension: 10_000_000_000,
            max_tensor_bytes: 500 * 1024 * 1024 * 1024,
            max_total_tensor_bytes: 1_000 * 1024 * 1024 * 1024,
            max_graph_nodes: 1_000_000,
            max_graph_edges: 5_000_000,
            max_nesting_depth: 128,
            max_string_bytes: 16 * 1024 * 1024,
            max_archive_entries: 500_000,
            max_archive_expanded_bytes: 1_000 * 1024 * 1024 * 1024,
        }
    }

    /// Verifies that a tensor shape's rank and total dimensions comply with limits.
    pub fn check_shape(&self, dims: &[usize]) -> Result<()> {
        if dims.len() > self.max_rank {
            return Err(Error::Msg(alloc::format!(
                "Tensor rank {} exceeds maximum limit {}",
                dims.len(),
                self.max_rank
            )));
        }
        for (idx, &d) in dims.iter().enumerate() {
            if u64::try_from(d).map_or(true, |dimension| dimension > self.max_dimension) {
                return Err(Error::Msg(alloc::format!(
                    "Tensor dimension at axis {} ({}) exceeds limit {}",
                    idx,
                    d,
                    self.max_dimension
                )));
            }
        }
        Ok(())
    }

    /// Verifies that header length is within limits.
    pub fn check_header_bytes(&self, bytes: u64) -> Result<()> {
        if bytes > self.max_header_bytes {
            return Err(Error::Msg(alloc::format!(
                "Header length {} bytes exceeds maximum limit {} bytes",
                bytes,
                self.max_header_bytes
            )));
        }
        Ok(())
    }
}
