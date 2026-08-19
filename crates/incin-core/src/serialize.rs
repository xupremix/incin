#[cfg(feature = "std")]
use crate::err::{Error, Result};
#[cfg(feature = "std")]
use crate::nn::{StatePath, StateRole, StateSnapshot, StateValue};
#[cfg(feature = "std")]
use crate::shapes::ShapeBuf;
#[cfg(feature = "std")]
use crate::tensor::backend::Backend;
#[cfg(feature = "std")]
use crate::tensor::dtype::{DTypeDescriptor, DTypeId};
#[cfg(feature = "std")]
use crate::tensor::prelude::Device;
#[cfg(feature = "std")]
use alloc::{collections::BTreeMap, string::String, vec::Vec};

#[cfg(feature = "std")]
fn safetensors_dtype(dtype: DTypeDescriptor) -> anyhow::Result<safetensors::tensor::Dtype> {
    use safetensors::tensor::Dtype;
    match dtype.builtin_id() {
        Some(DTypeId::F32) => Ok(Dtype::F32),
        Some(DTypeId::F64) => Ok(Dtype::F64),
        Some(DTypeId::F16) => Ok(Dtype::F16),
        Some(DTypeId::BF16) => Ok(Dtype::BF16),
        Some(DTypeId::U32) => Ok(Dtype::U32),
        Some(DTypeId::I64) => Ok(Dtype::I64),
        Some(DTypeId::U8) => Ok(Dtype::U8),
        Some(DTypeId::Bool) => Ok(Dtype::BOOL),
        _ => Err(anyhow::anyhow!(
            "unsupported safetensors dtype {}",
            dtype.name()
        )),
    }
}

#[cfg(feature = "std")]
fn dtype_from_safetensors(dtype: safetensors::tensor::Dtype) -> anyhow::Result<DTypeDescriptor> {
    Ok(match dtype {
        safetensors::tensor::Dtype::F32 => DTypeId::F32,
        safetensors::tensor::Dtype::F64 => DTypeId::F64,
        safetensors::tensor::Dtype::F16 => DTypeId::F16,
        safetensors::tensor::Dtype::BF16 => DTypeId::BF16,
        safetensors::tensor::Dtype::U32 => DTypeId::U32,
        safetensors::tensor::Dtype::I64 => DTypeId::I64,
        safetensors::tensor::Dtype::U8 => DTypeId::U8,
        safetensors::tensor::Dtype::BOOL => DTypeId::Bool,
        _ => return Err(anyhow::anyhow!("unsupported dtype in safetensors")),
    }
    .descriptor())
}

/// The schema version stamped into every state file this crate writes.
///
/// It describes the *envelope* — how paths, roles, dtypes, and payload bytes
/// are arranged — and is deliberately independent of the crate version and of
/// any individual dtype's own descriptor version. Bump it when a reader of the
/// previous version would misread a file rather than fail to parse it.
#[cfg(feature = "std")]
pub const STATE_FORMAT_VERSION: u32 = 1;

/// The safetensors metadata key carrying [`STATE_FORMAT_VERSION`].
///
/// Foreign safetensors files (a Hugging Face checkpoint, say) do not carry it.
/// That is the point: this key is what distinguishes a file this crate wrote,
/// whose role and dtype conventions the reader may assume, from one it did
/// not.
#[cfg(feature = "std")]
const STATE_FORMAT_VERSION_KEY: &str = "incin.format.version";

/// Accepts a version this build can read, and refuses one it cannot with a
/// message naming both numbers, so a user who meets a newer file learns which
/// version would read it rather than a parse error from the middle of a
/// payload.
#[cfg(feature = "std")]
fn accept_state_format_version(found: Option<u32>, format: &str) -> anyhow::Result<u32> {
    match found {
        Some(version) if version <= STATE_FORMAT_VERSION => Ok(version),
        Some(version) => Err(anyhow::anyhow!(
            "{format} state file declares format version {version}, but this build reads at most \
             version {STATE_FORMAT_VERSION}; upgrade incin to read it"
        )),
        None => Err(anyhow::anyhow!(
            "{format} state file carries no `{STATE_FORMAT_VERSION_KEY}`, so it was not written by \
             a versioned incin build; re-save it with this version"
        )),
    }
}

#[cfg(feature = "std")]
pub(crate) fn serialize_snapshot_safetensors(
    snapshot: &StateSnapshot,
    path: &std::path::Path,
) -> anyhow::Result<()> {
    use safetensors::tensor::TensorView;
    let mut storage = Vec::new();
    let mut views = BTreeMap::new();
    let mut metadata = std::collections::HashMap::new();
    metadata.insert(
        STATE_FORMAT_VERSION_KEY.to_string(),
        STATE_FORMAT_VERSION.to_string(),
    );
    for (name, value) in snapshot.iter() {
        metadata.insert(
            format!("incin.state.role.{}", name.as_str()),
            match value.role() {
                StateRole::Parameter => "parameter".to_string(),
                StateRole::Buffer => "buffer".to_string(),
            },
        );
        storage.push((
            name.as_str().to_owned(),
            value.bytes().to_vec(),
            value.shape().dims().to_vec(),
            safetensors_dtype(value.dtype())?,
        ));
    }
    for (name, bytes, shape, dtype) in &storage {
        views.insert(name.clone(), TensorView::new(*dtype, shape.clone(), bytes)?);
    }
    safetensors::tensor::serialize_to_file(&views, Some(metadata), path)?;
    Ok(())
}

#[cfg(feature = "std")]
pub(crate) fn deserialize_snapshot_safetensors(
    path: &std::path::Path,
) -> anyhow::Result<StateSnapshot> {
    let bytes = std::fs::read(path)?;
    let (_, header) = safetensors::SafeTensors::read_metadata(&bytes)?;
    let metadata = header.metadata().as_ref();
    let declared = metadata
        .and_then(|items| items.get(STATE_FORMAT_VERSION_KEY))
        .map(|raw| {
            raw.parse::<u32>().map_err(|_| {
                anyhow::anyhow!("safetensors state file has a non-numeric format version {raw:?}")
            })
        })
        .transpose()?;
    accept_state_format_version(declared, "safetensors")?;
    read_safetensors_entries(&bytes, metadata)
}

/// Parses a safetensors file into backend-neutral owned state, without
/// requiring an `incin.format.version` key.
///
/// [`deserialize_snapshot_safetensors`] is the loader behind `ModelExt::load`
/// and refuses an unversioned file, because that file was never written by a
/// versioned incin build and the version contract cannot say anything about
/// it. A file downloaded from an external source — the Hugging Face Hub,
/// most obviously — is unversioned for exactly that reason and is never
/// wrong to be so: it was written by whatever produced it, not by incin. This
/// entry point exists for that case: same tensor/shape/dtype parsing, same
/// `incin.state.role.<name>` lookup (defaulting to `Parameter` when the key
/// is absent, which it always is for a genuinely foreign file), no version
/// gate. `import_model!`'s compile-time safetensors reader already accepts
/// foreign files on this same basis; this is the runtime equivalent for
/// callers who only know the file at runtime (e.g. after downloading it).
#[cfg(feature = "std")]
pub(crate) fn deserialize_snapshot_safetensors_foreign(
    path: &std::path::Path,
) -> anyhow::Result<StateSnapshot> {
    let bytes = std::fs::read(path)?;
    let (_, header) = safetensors::SafeTensors::read_metadata(&bytes)?;
    read_safetensors_entries(&bytes, header.metadata().as_ref())
}

#[cfg(feature = "std")]
fn read_safetensors_entries(
    bytes: &[u8],
    metadata: Option<&std::collections::HashMap<String, String>>,
) -> anyhow::Result<StateSnapshot> {
    let tensors = safetensors::SafeTensors::deserialize(bytes)?;
    let mut snapshot = StateSnapshot::new();
    for (name, view) in tensors.tensors() {
        let role = match metadata.and_then(|items| items.get(&format!("incin.state.role.{}", name)))
        {
            Some(role_str) => match role_str.as_str() {
                "parameter" => StateRole::Parameter,
                "buffer" => StateRole::Buffer,
                other => anyhow::bail!("unknown state role {other:?} for entry {name}"),
            },
            None => StateRole::Parameter,
        };
        snapshot.insert(
            StatePath::new(name)?,
            StateValue::new(
                ShapeBuf::from_slice(view.shape()),
                dtype_from_safetensors(view.dtype())?,
                view.data().to_vec(),
                role,
            )?,
        )?;
    }
    Ok(snapshot)
}

#[cfg(feature = "std")]
pub(crate) fn serialize_snapshot_postcard(
    snapshot: &StateSnapshot,
    path: &std::path::Path,
) -> anyhow::Result<()> {
    let envelope = StateWireEnvelope {
        version: STATE_FORMAT_VERSION,
        entries: snapshot
            .iter()
            .map(|(path, value)| StateWireEntry {
                path: path.as_str().to_string(),
                shape: value.shape().dims().to_vec(),
                dtype: value.dtype(),
                bytes: value.bytes().to_vec(),
                role: value.role(),
            })
            .collect::<Vec<_>>(),
    };
    std::fs::write(path, postcard::to_stdvec(&envelope)?)?;
    Ok(())
}

#[cfg(feature = "std")]
pub(crate) fn deserialize_snapshot_postcard(
    path: &std::path::Path,
) -> anyhow::Result<StateSnapshot> {
    // A postcard payload is a bare byte sequence with no self-describing
    // header, so an unversioned file cannot be told apart from a versioned one
    // by inspection: it decodes as whatever the current struct says it is. The
    // version leads the envelope so a mismatch is reported here rather than as
    // a truncated payload several fields later.
    let envelope: StateWireEnvelope = postcard::from_bytes(&std::fs::read(path)?)
        .map_err(|error| anyhow::anyhow!("postcard state file is not a state envelope: {error}"))?;
    accept_state_format_version(Some(envelope.version), "postcard")?;
    let mut snapshot = StateSnapshot::new();
    for entry in envelope.entries {
        snapshot.insert(
            StatePath::new(entry.path)?,
            StateValue::new(
                ShapeBuf::from_slice(&entry.shape),
                entry.dtype,
                entry.bytes,
                entry.role,
            )?,
        )?;
    }
    Ok(snapshot)
}

/// The postcard payload's outermost record. `version` is first so a reader
/// meeting a newer file refuses on the number rather than on a field it cannot
/// interpret.
#[cfg(feature = "std")]
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct StateWireEnvelope {
    version: u32,
    entries: Vec<StateWireEntry>,
}

#[cfg(feature = "std")]
#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct StateWireEntry {
    path: String,
    shape: Vec<usize>,
    dtype: DTypeDescriptor,
    bytes: Vec<u8>,
    role: StateRole,
}

#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Safetensors,
    Postcard,
    ONNX,
}

#[cfg(feature = "std")]
pub trait ModelExt<B: Backend + crate::tensor::backend::VariableBackend> {
    fn save(&self, format: Format, path: &std::path::Path) -> Result<()>
    where
        <<B as crate::tensor::backend::StorageBackend>::Device as Device>::Field: Default;

    /// Restores state in place, leaving every parameter where it already
    /// lives. There is no device argument: `load` used to take one and ignore
    /// it, which read as a relocation the call never performed. Moving a model
    /// between devices is `ToDevice`, a separate and explicit operation.
    fn load(&mut self, format: Format, path: &std::path::Path) -> Result<()>
    where
        <<B as crate::tensor::backend::StorageBackend>::Device as Device>::Field: Default;
}

#[cfg(feature = "std")]
impl<
    B: Backend + crate::tensor::backend::VariableBackend,
    T: crate::nn::VisitState<B> + crate::nn::VisitStateMut<B>,
> ModelExt<B> for T
{
    fn save(&self, format: Format, path: &std::path::Path) -> Result<()>
    where
        <<B as crate::tensor::backend::StorageBackend>::Device as Device>::Field: Default,
    {
        match format {
            Format::Safetensors => {
                serialize_snapshot_safetensors(&crate::nn::collect_state::<B, _>(self)?, path)
            }
            Format::Postcard => {
                serialize_snapshot_postcard(&crate::nn::collect_state::<B, _>(self)?, path)
            }
            Format::ONNX => Err(anyhow::anyhow!("ONNX is not a state format")),
        }
        .map_err(|e| Error::Msg(e.to_string()))
    }

    fn load(&mut self, format: Format, path: &std::path::Path) -> Result<()>
    where
        <<B as crate::tensor::backend::StorageBackend>::Device as Device>::Field: Default,
    {
        let snapshot = match format {
            Format::Safetensors => deserialize_snapshot_safetensors(path),
            Format::Postcard => deserialize_snapshot_postcard(path),
            Format::ONNX => return Err(Error::Msg("ONNX is not a state format".into())),
        }
        .map_err(|e| Error::Msg(e.to_string()))?;
        crate::nn::load_state::<B, _>(self, &snapshot)
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;

    fn fixture() -> StateSnapshot {
        let mut snapshot = StateSnapshot::new();
        for (index, dtype) in [
            DTypeId::F32,
            DTypeId::F16,
            DTypeId::BF16,
            DTypeId::I64,
            DTypeId::U32,
            DTypeId::U8,
            DTypeId::Bool,
        ]
        .into_iter()
        .enumerate()
        {
            let descriptor = dtype.descriptor();
            let byte_len = descriptor
                .size_bytes(32, crate::shapes::error::OperationKind::Storage)
                .expect("fixture dtype has storage bytes");
            snapshot
                .insert(
                    StatePath::new(format!("entry_{index}")).expect("fixture path is canonical"),
                    StateValue::new(
                        ShapeBuf::from_slice(&[32]),
                        descriptor,
                        vec![index as u8; byte_len],
                        if index % 2 == 0 {
                            StateRole::Parameter
                        } else {
                            StateRole::Buffer
                        },
                    )
                    .expect("fixture value is valid"),
                )
                .expect("fixture paths are unique");
        }
        snapshot
    }

    #[test]
    fn safetensors_round_trips_exact_supported_native_dtypes() {
        let path = std::env::temp_dir().join(format!(
            "incin-state-serialize-{}.safetensors",
            std::process::id()
        ));
        let expected = fixture();
        serialize_snapshot_safetensors(&expected, &path).expect("serialize snapshot");
        let actual = deserialize_snapshot_safetensors(&path).expect("deserialize snapshot");
        assert_eq!(actual, expected);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn postcard_round_trips_exact_supported_native_dtypes() {
        let path = std::env::temp_dir().join(format!(
            "incin-state-serialize-{}.postcard",
            std::process::id()
        ));
        let expected = fixture();
        serialize_snapshot_postcard(&expected, &path).expect("serialize snapshot");
        let actual = deserialize_snapshot_postcard(&path).expect("deserialize snapshot");
        assert_eq!(actual, expected);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn postcard_supports_q8_and_safetensors_rejects_it_explicitly() {
        let descriptor = DTypeId::Q8_0.descriptor();
        let bytes = descriptor
            .size_bytes(32, crate::shapes::error::OperationKind::Storage)
            .expect("q8 fixture has storage bytes");
        let mut snapshot = StateSnapshot::new();
        snapshot
            .insert(
                StatePath::new("quantized").expect("canonical path"),
                StateValue::new(
                    ShapeBuf::from_slice(&[32]),
                    descriptor,
                    vec![0; bytes],
                    StateRole::Parameter,
                )
                .expect("q8 fixture is valid"),
            )
            .expect("unique path");
        let postcard_path =
            std::env::temp_dir().join(format!("incin-state-q8-{}.postcard", std::process::id()));
        serialize_snapshot_postcard(&snapshot, &postcard_path).expect("serialize q8");
        assert_eq!(
            deserialize_snapshot_postcard(&postcard_path).expect("deserialize q8"),
            snapshot
        );
        std::fs::remove_file(&postcard_path).ok();

        let safetensors_path =
            std::env::temp_dir().join(format!("incin-state-q8-{}.safetensors", std::process::id()));
        assert!(serialize_snapshot_safetensors(&snapshot, &safetensors_path).is_err());
        std::fs::remove_file(safetensors_path).ok();
    }

    /// A unique path per test, so the suite's threads cannot collide on one
    /// temporary file.
    fn scratch(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("incin-state-{name}-{}", std::process::id()))
    }

    #[test]
    fn every_written_safetensors_file_declares_the_state_format_version() {
        let path = scratch("version-stamp.safetensors");
        serialize_snapshot_safetensors(&fixture(), &path).expect("serialize snapshot");

        let bytes = std::fs::read(&path).expect("written file is readable");
        let (_, header) =
            safetensors::SafeTensors::read_metadata(&bytes).expect("written file has a header");
        assert_eq!(
            header
                .metadata()
                .as_ref()
                .and_then(|items| items.get(STATE_FORMAT_VERSION_KEY))
                .map(String::as_str),
            Some(STATE_FORMAT_VERSION.to_string().as_str()),
            "a state file without a version stamp cannot be told from a foreign safetensors file"
        );
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn a_safetensors_file_without_a_version_is_refused_as_unversioned() {
        // Written through safetensors directly, with role metadata but no
        // version key: exactly the shape of a file an unversioned build wrote,
        // and of a foreign checkpoint that never carried incin conventions.
        let path = scratch("unversioned.safetensors");
        let data = vec![0u8; 4];
        let view =
            safetensors::tensor::TensorView::new(safetensors::tensor::Dtype::F32, vec![1], &data)
                .expect("view is well formed");
        let mut views = BTreeMap::new();
        views.insert("entry_0".to_string(), view);
        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            "incin.state.role.entry_0".to_string(),
            "parameter".to_string(),
        );
        safetensors::tensor::serialize_to_file(&views, Some(metadata), &path)
            .expect("fixture file is written");

        let error = deserialize_snapshot_safetensors(&path)
            .expect_err("an unversioned state file must be refused")
            .to_string();
        assert!(
            error.contains(STATE_FORMAT_VERSION_KEY),
            "the refusal must name the missing key, got: {error}"
        );
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn the_foreign_loader_accepts_exactly_the_file_the_strict_loader_refuses() {
        // Same fixture shape as `a_safetensors_file_without_a_version_is_refused_as_unversioned`:
        // no version key, but with role metadata present, to prove the
        // foreign loader still reads role metadata when it happens to exist
        // rather than blindly defaulting every entry.
        let path = scratch("foreign-with-role.safetensors");
        let data = vec![0u8; 4];
        let view =
            safetensors::tensor::TensorView::new(safetensors::tensor::Dtype::F32, vec![1], &data)
                .expect("view is well formed");
        let mut views = BTreeMap::new();
        views.insert("entry_0".to_string(), view);
        let mut metadata = std::collections::HashMap::new();
        metadata.insert("incin.state.role.entry_0".to_string(), "buffer".to_string());
        safetensors::tensor::serialize_to_file(&views, Some(metadata), &path)
            .expect("fixture file is written");

        assert!(
            deserialize_snapshot_safetensors(&path).is_err(),
            "the strict loader must still refuse this file"
        );
        let snapshot = deserialize_snapshot_safetensors_foreign(&path)
            .expect("the foreign loader must accept a file with no version key");
        assert_eq!(snapshot.len(), 1);
        let (state_path, value) = snapshot.iter().next().expect("one entry");
        assert_eq!(state_path.as_str(), "entry_0");
        assert_eq!(
            value.role(),
            StateRole::Buffer,
            "role metadata is still honored when present, even without a version key"
        );
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn the_foreign_loader_defaults_role_to_parameter_when_absent() {
        // The realistic case: a genuinely third-party file with neither a
        // version key nor incin's role convention at all.
        let path = scratch("foreign-no-metadata.safetensors");
        let data = vec![0u8; 4];
        let view =
            safetensors::tensor::TensorView::new(safetensors::tensor::Dtype::F32, vec![1], &data)
                .expect("view is well formed");
        let mut views = BTreeMap::new();
        views.insert("weight".to_string(), view);
        safetensors::tensor::serialize_to_file(&views, None, &path)
            .expect("fixture file is written");

        let snapshot = deserialize_snapshot_safetensors_foreign(&path)
            .expect("a file with no incin metadata at all must still be readable");
        let (_, value) = snapshot.iter().next().expect("one entry");
        assert_eq!(value.role(), StateRole::Parameter);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn a_newer_state_version_is_refused_with_both_numbers_named() {
        let future = STATE_FORMAT_VERSION + 1;

        let error = accept_state_format_version(Some(future), "safetensors")
            .expect_err("a future version must be refused")
            .to_string();
        assert!(
            error.contains(&future.to_string())
                && error.contains(&STATE_FORMAT_VERSION.to_string()),
            "the refusal must name the file's version and this build's, got: {error}"
        );

        // The current version and every earlier one stay readable, which is
        // what makes the check a compatibility boundary rather than a pin.
        assert!(accept_state_format_version(Some(STATE_FORMAT_VERSION), "postcard").is_ok());
    }

    #[test]
    fn an_unknown_state_role_is_refused() {
        let temp = tempfile::NamedTempFile::new().expect("temp file");
        let path = temp.path().to_path_buf();
        let data = vec![1u8, 2, 3, 4];
        let view = safetensors::tensor::TensorView::new(safetensors::Dtype::U8, vec![1, 4], &data)
            .expect("tensor view is built");
        let mut views = BTreeMap::new();
        views.insert("entry_0".to_string(), view);
        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            STATE_FORMAT_VERSION_KEY.to_string(),
            STATE_FORMAT_VERSION.to_string(),
        );
        metadata.insert(
            "incin.state.role.entry_0".to_string(),
            "invalid_role".to_string(),
        );
        safetensors::tensor::serialize_to_file(&views, Some(metadata), &path)
            .expect("fixture file is written");

        let error = deserialize_snapshot_safetensors(&path)
            .expect_err("unknown state role must be refused")
            .to_string();
        assert!(error.contains("unknown state role"));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn a_postcard_file_carrying_a_newer_version_is_refused_before_its_payload() {
        // Serialized from the envelope struct with a bumped version, so the
        // entries after it are well formed. Only the version is wrong, which
        // is what proves the refusal came from the version check and not from
        // a decode failure further in.
        let path = scratch("future.postcard");
        let envelope = StateWireEnvelope {
            version: STATE_FORMAT_VERSION + 1,
            entries: Vec::new(),
        };
        std::fs::write(
            &path,
            postcard::to_stdvec(&envelope).expect("encode envelope"),
        )
        .expect("fixture file is written");

        let error = deserialize_snapshot_postcard(&path)
            .expect_err("a future postcard version must be refused")
            .to_string();
        assert!(
            error.contains("postcard") && error.contains(&(STATE_FORMAT_VERSION + 1).to_string()),
            "the refusal must name the format and the version, got: {error}"
        );
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn a_postcard_file_that_is_not_a_state_envelope_is_refused_by_name() {
        let path = scratch("garbage.postcard");
        std::fs::write(&path, [0xffu8; 32]).expect("fixture file is written");

        let error = deserialize_snapshot_postcard(&path)
            .expect_err("a non-envelope payload must be refused")
            .to_string();
        assert!(
            error.contains("not a state envelope"),
            "the refusal must say what the file is not, got: {error}"
        );
        std::fs::remove_file(path).ok();
    }
}
