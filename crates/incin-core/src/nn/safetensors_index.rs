//! Sharded Safetensors checkpoint indexes.
//!
//! A Hugging Face-style sharded checkpoint is a directory of ordinary
//! `.safetensors` files plus one `model.safetensors.index.json` whose
//! `weight_map` names the shard owning each tensor. This module is the
//! single reader for that layout: it parses and validates the index once,
//! resolves every shard path relative to the index itself, and hands out
//! tensor names in deterministic (sorted) order regardless of shard order
//! or JSON key order.

use crate::err::{Error, ErrorMessage, Result};
use alloc::{collections::BTreeMap, string::String, vec::Vec};use std::path::{Component, Path, PathBuf};

/// Operation tag used for structured errors raised while reading an index.
const OP: &str = "safetensors_index_open";

/// A validated view of one `*.safetensors.index.json`.
///
/// Construct with [`SafetensorsIndex::open`]. Construction performs every
/// structural check that does not require reading shard contents:
///
/// - duplicate keys anywhere in `weight_map` are rejected (a plain
///   `serde_json::Value` parse would silently keep the last one, which is
///   exactly the "duplicate tensor ownership" failure this reader exists to
///   catch);
/// - each mapped shard must be a bare file name, so resolution can never
///   escape the directory containing the index;
/// - every referenced shard must exist on disk at open time.
#[derive(Debug, Clone)]
pub struct SafetensorsIndex {
    /// Directory containing the index; all shards resolve inside it.
    root: PathBuf,
    /// Declared total byte size across shards, when the index carries one.
    total_size: Option<u64>,
    /// Tensor name to shard file name, sorted by tensor name.
    weight_map: BTreeMap<String, String>,
}

/// Collects `(tensor, shard)` pairs from a JSON object, rejecting duplicate
/// keys instead of silently keeping the last value.
struct WeightMapVisitor {
    pairs: Vec<(String, String)>,
}

impl<'de> serde::de::Visitor<'de> for WeightMapVisitor {
    type Value = Vec<(String, String)>;

    fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("a JSON object mapping tensor names to shard file names")
    }

    fn visit_map<A: serde::de::MapAccess<'de>>(
        mut self,
        mut map: A,
    ) -> std::result::Result<Self::Value, A::Error> {
        while let Some(name) = map.next_key::<String>()? {
            let shard: String = map.next_value()?;
            if self.pairs.iter().any(|(seen, _)| seen == &name) {
                return Err(serde::de::Error::custom(format!(
                    "duplicate entry for tensor `{name}`"
                )));
            }
            self.pairs.push((name, shard));
        }
        Ok(self.pairs)
    }
}

/// Reads the whole index document with duplicate-key detection for both the
/// top level and `weight_map`.
/// Parsed index document: declared total size and ordered weight-map pairs.
type ParsedIndex = (Option<u64>, Vec<(String, String)>);

fn parse_index_document(raw: &str) -> Result<ParsedIndex> {
    use serde::de::Deserialize;

    let mut de = serde_json::Deserializer::from_str(raw);
    let parsed: RawIndex = Deserialize::deserialize(&mut de)
        .map_err(|e| malformed(format!("invalid index JSON: {e}")))?;
    de.end()
        .map_err(|e| malformed(format!("trailing data after index JSON: {e}")))?;
    Ok((parsed.metadata.and_then(|m| m.total_size), parsed.weight_map))
}

/// Shape of the index document we accept: `metadata.total_size` optional,
/// `weight_map` required, unknown fields ignored so future format additions
/// do not strand older readers.
#[derive(serde::Deserialize)]
struct RawIndex {
    #[serde(default)]
    metadata: Option<RawIndexMetadata>,
    #[serde(deserialize_with = "reject_duplicate_weight_map")]
    weight_map: Vec<(String, String)>,
}

#[derive(serde::Deserialize)]
struct RawIndexMetadata {
    #[serde(default)]
    total_size: Option<u64>,
}

/// Deserializes `weight_map` as ordered pairs so duplicate tensor entries
/// surface as errors rather than last-writer-wins.
fn reject_duplicate_weight_map<'de, D>(de: D) -> std::result::Result<Vec<(String, String)>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    de.deserialize_map(WeightMapVisitor { pairs: Vec::new() })
}

fn malformed(reason: impl AsRef<str>) -> Error {
    Error::MalformedArtifact {
        operation: OP,
        artifact: "safetensors index",
        reason: ErrorMessage::new(reason.as_ref()),
    }
}

impl SafetensorsIndex {
    /// Opens and validates the index at `path`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let raw = std::fs::read_to_string(path).map_err(|e| {
            malformed(format!("reading safetensors index {}: {e}", path.display()))
        })?;
        // Reject traversal-shaped documents before touching the filesystem.
        let (total_size, weight_pairs) = parse_index_document(&raw)?;

        let root = path
            .parent()
            .ok_or_else(|| malformed(format!("index path {} has no parent directory", path.display())))?
            .to_path_buf();

        let mut weight_map = BTreeMap::new();
        for (name, shard) in weight_pairs {
            validate_shard_name(&shard)?;
            if weight_map.insert(name.clone(), shard.clone()).is_some() {
                // Unreachable through `reject_duplicate_weight_map`, kept as a
                // structural guarantee on the type invariant.
                return Err(malformed(format!("duplicate entry for tensor `{name}`")));
            }
        }

        let index = Self {
            root: root.to_path_buf(),
            total_size,
            weight_map,
        };

        // Every referenced shard must exist now, not lazily mid-load.
        for shard in index.shards() {
            let path = index.shard_path(shard);
            if !path.is_file() {
                return Err(malformed(format!(
                    "shard `{shard}` referenced by the index is missing at {}",
                    path.display()
                )));
            }
        }
        Ok(index)
    }

    /// The directory all shard paths resolve into.
    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Declared total byte size across shards, when present.
    pub fn total_size(&self) -> Option<u64> {
        self.total_size
    }

    /// Number of mapped tensors.
    pub fn len(&self) -> usize {
        self.weight_map.len()
    }

    /// Whether the map claims no tensors.
    pub fn is_empty(&self) -> bool {
        self.weight_map.is_empty()
    }

    /// Iterates `(tensor name, shard name)` in sorted tensor-name order.
    pub fn tensors(&self) -> impl ExactSizeIterator<Item = (&str, &str)> + '_ {
        self.weight_map.iter().map(|(n, s)| (n.as_str(), s.as_str()))
    }

    /// The shard that owns `tensor`, if the map names it.
    pub fn shard_of(&self, tensor: &str) -> Option<&str> {
        self.weight_map.get(tensor).map(String::as_str)
    }

    /// Every distinct shard name, sorted.
    pub fn shards(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.weight_map.values().map(String::as_str).collect();
        names.sort_unstable();
        names.dedup();
        names
    }

    /// Resolves `shard` against the index directory. The name was validated
    /// at open time, so this cannot escape the checkpoint root.
    pub fn shard_path(&self, shard: &str) -> PathBuf {
        self.root.join(shard)
    }

    /// Reads one shard's bytes.
    pub fn read_shard(&self, shard: &str) -> Result<Vec<u8>> {
        std::fs::read(self.shard_path(shard))
            .map_err(|e| Error::Msg(format!("reading shard `{shard}` failed: {e}")))
    }
}

/// A shard reference must be a bare relative file name: no separators, no
/// parent/current-directory components, no absolute paths. This is what makes
/// [`SafetensorsIndex::shard_path`] structurally unable to leave the
/// checkpoint root.
fn validate_shard_name(shard: &str) -> Result<()> {
    if shard.is_empty() {
        return Err(malformed("weight_map maps a tensor to an empty shard name"));
    }
    let path = Path::new(shard);
    if path.is_absolute() {
        return Err(malformed(format!(
            "weight_map maps a tensor to absolute shard path `{shard}`"
        )));
    }
    if !path.components().all(|c| matches!(c, Component::Normal(_))) {
        return Err(malformed(format!(
            "weight_map shard reference `{shard}` is not a bare file name under the checkpoint root"
        )));
    }
    Ok(())
}


#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::nn::StateRole;
    use crate::serialize::deserialize_snapshot_safetensors_index;
    use std::collections::BTreeMap;

    /// Writes one shard file and returns its byte length.
    fn write_shard(
        dir: &Path,
        file: &str,
        tensors: &[(&str, Vec<f32>)],
        roles: &[(&str, &str)],
    ) -> u64 {
        let raw: Vec<Vec<u8>> = tensors
            .iter()
            .map(|(_, values)| values.iter().flat_map(|v| v.to_le_bytes()).collect())
            .collect();
        let mut views = BTreeMap::new();
        for ((name, values), bytes) in tensors.iter().zip(&raw) {
            let view = safetensors::tensor::TensorView::new(
                safetensors::tensor::Dtype::F32,
                vec![values.len()],
                bytes,
            )
            .expect("view is well formed");
            views.insert((*name).to_string(), view);
        }
        let metadata: std::collections::HashMap<String, String> = roles
            .iter()
            .map(|(name, role)| (format!("incin.state.role.{name}"), (*role).to_string()))
            .collect();
        let path = dir.join(file);
        safetensors::tensor::serialize_to_file(&views, Some(metadata), &path)
            .expect("shard fixture is written");
        std::fs::metadata(&path).expect("shard exists").len()
    }

    fn write_index(dir: &Path, body: &str) -> PathBuf {
        let path = dir.join("model.safetensors.index.json");
        std::fs::write(&path, body).expect("index fixture is written");
        path
    }

    fn scratch(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("incin-index-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("scratch dir is created");
        dir
    }

    #[test]
    fn a_two_shard_checkpoint_loads_as_one_dictionary_in_sorted_order() {
        let dir = scratch("two-shard");
        let len_a = write_shard(
            &dir,
            "model-00001.safetensors",
            &[("layer.w", vec![1.0])],
            &[("layer.w", "parameter")],
        );
        let len_b = write_shard(
            &dir,
            "model-00002.safetensors",
            &[("head.bias", vec![2.0, 3.0]), ("head.norm", vec![4.0])],
            &[("head.norm", "buffer")],
        );
        let index_path = write_index(
            &dir,
            &format!(
                r#"{{"metadata": {{"total_size": {}}}, "weight_map": {{
                    "head.norm": "model-00002.safetensors",
                    "layer.w": "model-00001.safetensors",
                    "head.bias": "model-00002.safetensors"
                }}}}"#,
                len_a + len_b
            ),
        );

        let snapshot =
            deserialize_snapshot_safetensors_index(&index_path).expect("logical load succeeds");
        let names: Vec<&str> = snapshot.iter().map(|(path, _)| path.as_str()).collect();
        assert_eq!(names, vec!["head.bias", "head.norm", "layer.w"]);
        assert_eq!(snapshot.iter().nth(1).unwrap().1.role(), StateRole::Buffer);
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_declared_total_size_that_disagrees_with_disk_is_rejected() {
        let dir = scratch("total-size");
        write_shard(&dir, "a.safetensors", &[("x", vec![1.0])], &[]);
        let index_path = write_index(
            &dir,
            r#"{"metadata": {"total_size": 999999}, "weight_map": {"x": "a.safetensors"}}"#,
        );
        let error = deserialize_snapshot_safetensors_index(&index_path)
            .expect_err("declared size must match the shards")
            .to_string();
        assert!(error.contains("total_size"), "{error}");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn duplicate_weight_map_entries_are_rejected_by_name() {
        let dir = scratch("dup");
        write_shard(&dir, "a.safetensors", &[("x", vec![1.0])], &[]);
        let index_path = write_index(
            &dir,
            r#"{"weight_map": {"x": "a.safetensors", "x": "a.safetensors"}}"#,
        );
        let error = SafetensorsIndex::open(&index_path)
            .err()
            .map(|e| e.to_string())
            .unwrap_or_else(|| {
                deserialize_snapshot_safetensors_index(&index_path)
                    .expect_err("duplicate keys must be rejected")
                    .to_string()
            });
        assert!(error.contains("`x`"), "{error}");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn shard_references_may_not_traverse_out_of_the_checkpoint_root() {
        let dir = scratch("traversal");
        let outside = std::env::temp_dir().join(format!("incin-escape-{}", std::process::id()));
        std::fs::create_dir_all(&outside).expect("outside dir is created");
        write_shard(&outside, "escaped.safetensors", &[("x", vec![1.0])], &[]);
        let index_path =
            write_index(&dir, r#"{"weight_map": {"x": "../escaped.safetensors"}}"#);
        let error = deserialize_snapshot_safetensors_index(&index_path)
            .expect_err("traversal must be rejected")
            .to_string();
        assert!(error.contains("bare file name"), "{error}");
        std::fs::remove_dir_all(dir).ok();
        std::fs::remove_dir_all(outside).ok();
    }

    #[test]
    fn a_missing_shard_is_reported_by_name_at_open_time() {
        let dir = scratch("missing");
        write_shard(&dir, "present.safetensors", &[("x", vec![1.0])], &[]);
        let index_path = write_index(
            &dir,
            r#"{"weight_map": {"x": "present.safetensors", "y": "absent.safetensors"}}"#,
        );
        let error = SafetensorsIndex::open(&index_path)
            .expect_err("the missing shard must be rejected at open")
            .to_string();
        assert!(error.contains("absent.safetensors"), "{error}");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_shard_containing_an_unmapped_tensor_is_rejected_by_tensor_and_shard() {
        let dir = scratch("unmapped");
        write_shard(
            &dir,
            "a.safetensors",
            &[("x", vec![1.0]), ("stowaway", vec![2.0])],
            &[],
        );
        let index_path = write_index(&dir, r#"{"weight_map": {"x": "a.safetensors"}}"#);
        let error = deserialize_snapshot_safetensors_index(&index_path)
            .expect_err("an unmapped tensor must be rejected")
            .to_string();
        assert!(error.contains("stowaway") && error.contains("a.safetensors"), "{error}");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn a_tensor_mapped_to_one_shard_but_present_in_two_is_rejected() {
        let dir = scratch("dual-owner");
        write_shard(&dir, "a.safetensors", &[("x", vec![1.0])], &[]);
        write_shard(&dir, "b.safetensors", &[("x", vec![9.0])], &[]);
        let index_path = write_index(&dir, r#"{"weight_map": {"x": "a.safetensors"}}"#);
        let error = deserialize_snapshot_safetensors_index(&index_path)
            .expect_err("duplicate ownership must be rejected")
            .to_string();
        assert!(error.contains("`x`") && error.contains("b.safetensors"), "{error}");
        std::fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn an_index_claim_its_shard_does_not_contain_is_rejected() {
        let dir = scratch("ghost-claim");
        write_shard(&dir, "a.safetensors", &[("x", vec![1.0])], &[]);
        let index_path = write_index(
            &dir,
            r#"{"weight_map": {"x": "a.safetensors", "ghost": "a.safetensors"}}"#,
        );
        let error = deserialize_snapshot_safetensors_index(&index_path)
            .expect_err("an unsatisfied claim must be rejected")
            .to_string();
        assert!(error.contains("ghost") && error.contains("does not contain"), "{error}");
        std::fs::remove_dir_all(dir).ok();
    }
}
