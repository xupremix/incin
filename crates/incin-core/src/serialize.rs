#[cfg(feature = "std")]
use crate::err::{Error, ErrorMessage, Result};
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
/// It describes the *envelope* - how paths, roles, dtypes, and payload bytes
/// are arranged - and is deliberately independent of the crate version and of
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
/// A `.json` path names a sharded-checkpoint index (`*.safetensors.index.json`
/// by convention); anything else is a single safetensors file.
fn looks_like_safetensors_index(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| extension.eq_ignore_ascii_case("json"))
}

// ---------------------------------------------------------------------------
// Streaming safetensors access.
//
// A safetensors file is an 8-byte little-endian header length, the header JSON,
// then raw tensor bytes in header-declared ranges. Nothing in the format
// requires the payload to be resident to know where anything lives, so every
// loader below reads prefix and header first, validates the declared layout,
// and then reads one tensor's byte range at a time. Peak host residency for
// checkpoint bytes is therefore bounded by the larger of (header length,
// largest single tensor) rather than by checkpoint size, and at most one
// shard file is ever open.
// ---------------------------------------------------------------------------

/// Operation name for header/structure failures raised while opening a
/// safetensors stream.
#[cfg(feature = "std")]
const OP_OPEN_STREAM: &str = "open safetensors stream";

/// Operation name for index-level checks (`total_size`, shard cross-checks,
/// the unreferenced-sibling scan).
#[cfg(feature = "std")]
const OP_LOAD_INDEX: &str = "load safetensors index";

/// Operation name for payload-byte reads and per-tensor value validation.
#[cfg(feature = "std")]
const OP_READ_TENSOR: &str = "read safetensors tensor";

/// Mirror of safetensors's private `MAX_HEADER_SIZE`. The upstream crate caps
/// header JSON at 100 MB and refuses anything larger before parsing; a reader
/// that parses the header itself must apply the same cap or a hostile file
/// could make it allocate without bound.
#[cfg(feature = "std")]
const MAX_SAFETENSORS_HEADER_BYTES: u64 = 100_000_000;

/// The fixed-width little-endian prefix every safetensors file starts with.
#[cfg(feature = "std")]
const SAFETENSORS_PREFIX_LEN: usize = 8;

#[cfg(feature = "std")]
fn malformed_open(reason: impl AsRef<str>) -> Error {
    Error::MalformedArtifact {
        operation: OP_OPEN_STREAM,
        artifact: "safetensors file",
        reason: ErrorMessage::new(reason),
    }
}

#[cfg(feature = "std")]
fn malformed_shard(reason: impl AsRef<str>) -> Error {
    Error::MalformedArtifact {
        operation: OP_LOAD_INDEX,
        artifact: "safetensors shard",
        reason: ErrorMessage::new(reason),
    }
}

#[cfg(feature = "std")]
fn malformed_index(reason: impl AsRef<str>) -> Error {
    Error::MalformedArtifact {
        operation: OP_LOAD_INDEX,
        artifact: "safetensors index",
        reason: ErrorMessage::new(reason),
    }
}

#[cfg(feature = "std")]
fn io_open(message: impl AsRef<str>) -> Error {
    Error::Io {
        operation: OP_OPEN_STREAM,
        message: ErrorMessage::new(message),
    }
}

#[cfg(feature = "std")]
fn io_index(message: impl AsRef<str>) -> Error {
    Error::Io {
        operation: OP_LOAD_INDEX,
        message: ErrorMessage::new(message),
    }
}

#[cfg(feature = "std")]
fn io_read(message: impl AsRef<str>) -> Error {
    Error::Io {
        operation: OP_READ_TENSOR,
        message: ErrorMessage::new(message),
    }
}

/// Random-access byte reads over one checkpoint file.
///
/// Implementations return exactly `len` bytes or an `io::Error`; they never
/// return a short buffer. The streaming loaders ask for the header and then
/// one tensor range at a time, so a [`ByteSource`] is also the seam where a
/// test can measure how large any single read ever gets and how many files
/// are open at once, without instrumenting the filesystem itself.
#[cfg(feature = "std")]
pub(crate) trait ByteSource {
    /// Total length of the underlying file, captured when it was opened.
    fn len(&self) -> u64;

    /// Reads exactly `len` bytes starting at `offset`.
    fn read_exact_at(&mut self, offset: u64, len: usize) -> std::io::Result<Vec<u8>>;
}

/// A [`ByteSource`] over one file, opened once and seeked per read.
#[cfg(feature = "std")]
pub(crate) struct FileByteSource {
    file: std::fs::File,
    len: u64,
}

#[cfg(feature = "std")]
impl FileByteSource {
    /// Opens `path` and records its length, so later reads need no stat.
    pub(crate) fn open(path: &std::path::Path) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(|e| {
            io_open(format!(
                "opening safetensors file {} failed: {e}",
                path.display()
            ))
        })?;
        let len = file
            .metadata()
            .map_err(|e| {
                io_open(format!(
                    "statting safetensors file {} failed: {e}",
                    path.display()
                ))
            })?
            .len();
        Ok(Self { file, len })
    }
}

#[cfg(feature = "std")]
impl ByteSource for FileByteSource {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_exact_at(&mut self, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
        use std::io::{Read, Seek, SeekFrom};
        self.file.seek(SeekFrom::Start(offset))?;
        let mut buffer = vec![0u8; len];
        self.file.read_exact(&mut buffer)?;
        Ok(buffer)
    }
}

/// Opens a file through the default [`FileByteSource`].
#[cfg(feature = "std")]
fn open_file_source(path: &std::path::Path) -> Result<FileByteSource> {
    FileByteSource::open(path)
}

/// A safetensors header read far enough to place its tensors: the validated
/// [`safetensors::tensor::Metadata`] (offsets, sizes, and dtype/shape
/// agreement are all proven by its `Deserialize` impl) and the byte offset
/// where the data section begins.
#[cfg(feature = "std")]
struct ParsedHeader {
    header: safetensors::tensor::Metadata,
    data_start: u64,
}

/// Reads only the prefix and header JSON of one safetensors file.
///
/// `subject` is a human label for the file - ``shard `model-00001-of-00002.safetensors` ``
/// or ``file `model.safetensors` `` - and is embedded in every error so a
/// malformed header always names the file it came from.
///
/// This deliberately does *not* call `SafeTensors::read_metadata`: that
/// function requires the full file in memory to compare the declared layout
/// against a buffer length, which is exactly the whole-file residency this
/// loader exists to avoid. The same end-condition - header bytes plus declared
/// data bytes must equal the real file length - is re-derived here from the
/// file's own length.
#[cfg(feature = "std")]
fn read_safetensors_header<S: ByteSource>(source: &mut S, subject: &str) -> Result<ParsedHeader> {
    let file_len = source.len();
    let prefix = source
        .read_exact_at(0, SAFETENSORS_PREFIX_LEN)
        .map_err(|e| {
            io_open(format!(
                "reading the safetensors header of {subject} failed: {e}"
            ))
        })?;
    // `read_exact_at`'s contract is exactly `len` bytes or an error, so the
    // slice conversion cannot fail; it is the length field the caller asked
    // for, validated by that contract.
    let header_len_u64 = u64::from_le_bytes(
        prefix[..SAFETENSORS_PREFIX_LEN]
            .try_into()
            .expect("read_exact_at returned exactly the requested prefix length"),
    );
    if header_len_u64 > MAX_SAFETENSORS_HEADER_BYTES {
        return Err(malformed_open(format!(
            "{subject} declares a {header_len_u64}-byte header; safetensors headers are capped at \
             {MAX_SAFETENSORS_HEADER_BYTES} bytes"
        )));
    }
    let header_len = usize::try_from(header_len_u64).map_err(|_| {
        malformed_open(format!(
            "{subject}: header length {header_len_u64} does not fit this platform"
        ))
    })?;
    let header_bytes = source
        .read_exact_at(SAFETENSORS_PREFIX_LEN as u64, header_len)
        .map_err(|e| {
            io_open(format!(
                "reading the safetensors header of {subject} failed: {e}"
            ))
        })?;
    let text = core::str::from_utf8(&header_bytes).map_err(|e| {
        malformed_open(format!(
            "the safetensors header of {subject} is not valid UTF-8: {e}"
        ))
    })?;
    let header: safetensors::tensor::Metadata = serde_json::from_str(text).map_err(|e| {
        malformed_open(format!(
            "the safetensors header of {subject} is invalid: {e}"
        ))
    })?;
    let data_start = (SAFETENSORS_PREFIX_LEN as u64)
        .checked_add(header_len as u64)
        .ok_or_else(|| {
            malformed_open(format!(
                "the safetensors header of {subject} overflows this platform"
            ))
        })?;
    let data_len = header.data_len() as u64;
    let expected = data_start.checked_add(data_len).ok_or_else(|| {
        malformed_open(format!(
            "the safetensors header of {subject} declares an overflowing data length"
        ))
    })?;
    if expected != file_len {
        return Err(malformed_open(format!(
            "the safetensors header of {subject} declares {expected} total bytes ({data_start} \
             header + {data_len} data) but the file is {file_len} bytes"
        )));
    }
    Ok(ParsedHeader { header, data_start })
}

/// Resolves a tensor's state role from `incin.state.role.<name>` header
/// metadata, defaulting to [`StateRole::Parameter`] when the key is absent -
/// which it always is for a genuinely foreign file.
///
/// The `Err` string is the historical message (`unknown state role ... for
/// entry ...`); the caller wraps it with the file's subject label.
#[cfg(feature = "std")]
fn state_role_for(
    header: &safetensors::tensor::Metadata,
    name: &str,
) -> core::result::Result<StateRole, String> {
    match header
        .metadata()
        .as_ref()
        .and_then(|items| items.get(&format!("incin.state.role.{name}")))
    {
        Some(role_str) => match role_str.as_str() {
            "parameter" => Ok(StateRole::Parameter),
            "buffer" => Ok(StateRole::Buffer),
            other => Err(format!("unknown state role {other:?} for entry {name}")),
        },
        None => Ok(StateRole::Parameter),
    }
}

/// Reads `incin.format.version` from a header, if the file carries one.
#[cfg(feature = "std")]
fn declared_state_format_version(
    header: &safetensors::tensor::Metadata,
) -> core::result::Result<Option<u32>, String> {
    header
        .metadata()
        .as_ref()
        .and_then(|items| items.get(STATE_FORMAT_VERSION_KEY))
        .map(|raw| {
            raw.parse::<u32>().map_err(|_| {
                format!("safetensors state file has a non-numeric format version {raw:?}")
            })
        })
        .transpose()
}

/// Applies the strict single-file version contract of [`ModelExt::load`]: a
/// file that did not come from a versioned incin build cannot be assumed to
/// follow incin's role and dtype conventions, so it is refused by name before
/// any tensor is read.
#[cfg(feature = "std")]
fn require_state_format_version(header: &safetensors::tensor::Metadata) -> Result<()> {
    let declared = declared_state_format_version(header).map_err(malformed_open)?;
    accept_state_format_version(declared, "safetensors")
        .map_err(|e| malformed_open(e.to_string()))?;
    Ok(())
}

/// When a single safetensors file must prove it was written by a versioned
/// incin build before incin assumes the file follows incin conventions.
#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VersionGate {
    /// The `ModelExt::load` contract: version required.
    StrictInc,
    /// Runtime foreign-file loading (the Hugging Face Hub case): no version
    /// requirement, roles honored when present.
    Foreign,
}

/// One tensor's placement inside its shard: everything needed to read it by
/// byte range without parsing the header again.
#[cfg(feature = "std")]
struct TensorPlan {
    dtype: DTypeDescriptor,
    shape: Vec<usize>,
    /// Byte range relative to the shard's data-section start.
    range: (u64, u64),
    role: StateRole,
}

/// One file's validated, payload-free layout. Building this proves roles,
/// dtypes, shapes, and offsets are well formed; reading [`TensorPlan`]s from
/// it never consults the header again.
#[cfg(feature = "std")]
struct ShardLayout {
    /// Human label naming the file inside errors.
    subject: String,
    path: std::path::PathBuf,
    /// Byte offset where this file's data section begins.
    data_start: u64,
    /// Tensor plans in sorted-name order, so payload reads are deterministic.
    tensors: BTreeMap<String, TensorPlan>,
}

/// Turns a parsed header into per-tensor plans, validating every role and
/// dtype now - before any payload byte is read, and therefore before any
/// live module state could be touched.
#[cfg(feature = "std")]
fn build_shard_layout(
    subject: &str,
    path: std::path::PathBuf,
    parsed: ParsedHeader,
) -> Result<ShardLayout> {
    let header_tensors = parsed.header.tensors();
    let mut names: Vec<&String> = header_tensors.keys().collect();
    names.sort_unstable();
    let mut tensors = BTreeMap::new();
    for name in names {
        let info = header_tensors[name];
        let role = state_role_for(&parsed.header, name)
            .map_err(|reason| malformed_open(format!("{subject}: {reason}")))?;
        let dtype = dtype_from_safetensors(info.dtype)
            .map_err(|e| malformed_open(format!("{subject}: tensor `{name}`: {e}")))?;
        tensors.insert(
            name.clone(),
            TensorPlan {
                dtype,
                shape: info.shape.clone(),
                range: (info.data_offsets.0 as u64, info.data_offsets.1 as u64),
                role,
            },
        );
    }
    Ok(ShardLayout {
        subject: subject.to_string(),
        path,
        data_start: parsed.data_start,
        tensors,
    })
}

/// Proves one shard's header and the index agree, name for name.
///
/// Both directions matter: a shard carrying a tensor the index does not map
/// (or maps to a different shard) would let a later consumer see a file the
/// index never mentions, and a tensor the index maps but the shard lacks
/// would silently disappear. Historical message wording is preserved - these
/// strings are part of the loader's tested contract.
#[cfg(feature = "std")]
fn cross_check_index_shard(
    index: &crate::nn::safetensors_index::SafetensorsIndex,
    shard: &str,
    parsed: &ParsedHeader,
) -> Result<()> {
    let header_tensors = parsed.header.tensors();
    let mut names: Vec<&String> = header_tensors.keys().collect();
    names.sort_unstable();
    for name in names {
        match index.shard_of(name) {
            None => {
                return Err(malformed_shard(format!(
                    "shard `{shard}` contains tensor `{name}`, which the index does not map"
                )));
            }
            Some(owner) if owner != shard => {
                return Err(malformed_shard(format!(
                    "tensor `{name}` is mapped to shard `{owner}` but also appears in shard \
                     `{shard}`"
                )));
            }
            Some(_) => {}
        }
    }
    for (name, owner) in index.tensors() {
        if owner == shard && !header_tensors.contains_key(name) {
            return Err(malformed_shard(format!(
                "the index maps tensor `{name}` to shard `{shard}`, which does not contain it"
            )));
        }
    }
    Ok(())
}

/// Re-derives the index's optional `total_size` claim from the filesystem.
///
/// This stats without reading: byte residency is untouched, and the check
/// keeps its historical messages (`statting shard ...`, `total_size`).
#[cfg(feature = "std")]
fn verify_total_size(index: &crate::nn::safetensors_index::SafetensorsIndex) -> Result<()> {
    let Some(declared) = index.total_size() else {
        return Ok(());
    };
    let mut actual = 0u64;
    for shard in index.shards() {
        let len = std::fs::metadata(index.shard_path(shard))
            .map_err(|e| io_index(format!("statting shard `{shard}` failed: {e}")))?
            .len();
        actual = actual.saturating_add(len);
    }
    if actual != declared {
        return Err(malformed_index(format!(
            "index declares total_size {declared} but its shards total {actual} bytes"
        )));
    }
    Ok(())
}

/// Scans `.safetensors` files in the checkpoint directory that the index
/// never references, refusing any that duplicates a mapped tensor.
///
/// Duplicate ownership also hides outside the map: a sibling shard the index
/// never references can carry a copy of a mapped tensor, and whichever file a
/// later consumer happens to open would silently win. Only headers are read -
/// names are all this check needs - so the scan stays inside the same
/// bounded-residency contract as the rest of the loader.
#[cfg(feature = "std")]
fn scan_unreferenced_shards<S: ByteSource>(
    index: &crate::nn::safetensors_index::SafetensorsIndex,
    open: &impl Fn(&std::path::Path) -> Result<S>,
) -> Result<()> {
    let mapped: std::collections::BTreeSet<&str> = index.shards().into_iter().collect();
    let entries = std::fs::read_dir(index.root())
        .map_err(|e| io_index(format!("listing the checkpoint directory failed: {e}")))?;
    for entry in entries {
        let path = entry
            .map_err(|e| io_index(format!("listing the checkpoint directory failed: {e}")))?
            .path();
        let is_shard_file = path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("safetensors"));
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .map(alloc::string::ToString::to_string);
        let Some(file_name) = file_name else { continue };
        if !is_shard_file || mapped.contains(file_name.as_str()) {
            continue;
        }
        let mut source = open(&path).map_err(|e| {
            io_index(format!(
                "reading unreferenced shard `{file_name}` failed: {e}"
            ))
        })?;
        let subject = format!("unreferenced shard `{file_name}`");
        let parsed = read_safetensors_header(&mut source, &subject)?;
        for name in parsed.header.tensors().keys() {
            if let Some(owner) = index.shard_of(name) {
                return Err(malformed_shard(format!(
                    "tensor `{name}` is mapped to shard `{owner}` but also appears in unreferenced \
                     shard `{file_name}`"
                )));
            }
        }
    }
    Ok(())
}

/// Reads one planned tensor's bytes from its (already open) source.
///
/// The range was proven to fit the file when the header was accepted
/// (`data_start + data_len == file_len`), so an error here means the file
/// changed on disk or the device failed mid-read - never that the header
/// lied. Either way the failure names the tensor, the shard, the offset, and
/// the length it wanted.
#[cfg(feature = "std")]
fn read_tensor_bytes<S: ByteSource>(
    source: &mut S,
    layout: &ShardLayout,
    name: &str,
    plan: &TensorPlan,
) -> Result<Vec<u8>> {
    // `Metadata::validate` proved end >= start at header-accept time.
    let len_u64 = plan.range.1 - plan.range.0;
    let len = usize::try_from(len_u64).map_err(|_| {
        malformed_open(format!(
            "tensor `{name}` in {} has a {len_u64}-byte payload that does not fit this platform",
            layout.subject
        ))
    })?;
    // Both addends are bounded by the file length the header check proved, so
    // the sum cannot overflow.
    let offset = layout.data_start + plan.range.0;
    source.read_exact_at(offset, len).map_err(|e| {
        io_read(format!(
            "reading tensor `{name}` from {} at offset {offset} ({len} bytes) failed: {e}",
            layout.subject
        ))
    })
}

/// Wraps raw bytes into a validated [`StateValue`], naming the tensor and
/// file when the payload disagrees with its own header.
#[cfg(feature = "std")]
fn build_state_value(
    name: &str,
    subject: &str,
    plan: &TensorPlan,
    bytes: Vec<u8>,
) -> Result<StateValue> {
    StateValue::new(
        ShapeBuf::from_slice(&plan.shape),
        plan.dtype,
        bytes,
        plan.role,
    )
    .map_err(|e| Error::MalformedArtifact {
        operation: OP_READ_TENSOR,
        artifact: "safetensors tensor",
        reason: ErrorMessage::new(format!("tensor `{name}` in {subject}: {e}")),
    })
}

/// Streams every tensor of one layout into `snapshot`, one byte range at a
/// time, in sorted-name order. The source stays open across the loop and is
/// dropped by the caller, so a shard costs exactly one open for both its
/// header and all of its payloads.
#[cfg(feature = "std")]
fn stream_layout_payloads_into<S: ByteSource>(
    source: &mut S,
    layout: &ShardLayout,
    snapshot: &mut StateSnapshot,
) -> Result<()> {
    for (name, plan) in &layout.tensors {
        let bytes = read_tensor_bytes(source, layout, name, plan)?;
        let value = build_state_value(name, &layout.subject, plan, bytes)?;
        snapshot.insert(StatePath::new(name)?, value)?;
    }
    Ok(())
}

/// Loads a sharded checkpoint through its validated
/// [`SafetensorsIndex`](crate::nn::safetensors_index::SafetensorsIndex) as one
/// logical state dictionary.
///
/// Shard iteration order is sorted shard name, and the final key order of the
/// snapshot is deterministic because `StateSnapshot` is order-independent.
/// Structural honesty checks mirror the index contract: a shard containing a
/// tensor the index does not map, a tensor mapped to two shards, or an index
/// claim its shard does not contain are all errors naming the tensor and
/// shard. Sharded checkpoints are external artifacts by construction, so
/// entries load under foreign-file semantics (no `incin.format.version`
/// requirement).
///
/// Each shard is opened once: its header is validated and cross-checked
/// against the index, then its tensors stream into the snapshot one byte
/// range at a time, then it is closed before the next opens.
#[cfg(feature = "std")]
pub(crate) fn deserialize_snapshot_safetensors_index(
    path: &std::path::Path,
) -> anyhow::Result<StateSnapshot> {
    deserialize_snapshot_safetensors_index_via(path, open_file_source)
}

/// Injectable-opener form of [`deserialize_snapshot_safetensors_index`],
/// identical in behavior. Tests wrap `open` to instrument how large a single
/// read ever gets and how many files are open at once.
#[cfg(feature = "std")]
pub(crate) fn deserialize_snapshot_safetensors_index_via<S, F>(
    path: &std::path::Path,
    open: F,
) -> anyhow::Result<StateSnapshot>
where
    S: ByteSource,
    F: Fn(&std::path::Path) -> Result<S>,
{
    use crate::nn::safetensors_index::SafetensorsIndex;

    let index = SafetensorsIndex::open(path)?;
    verify_total_size(&index)?;
    let mut snapshot = StateSnapshot::new();
    for shard in index.shards() {
        let shard_path = index.shard_path(shard);
        let mut source = open(&shard_path)?;
        let subject = format!("shard `{shard}`");
        let parsed = read_safetensors_header(&mut source, &subject)?;
        cross_check_index_shard(&index, shard, &parsed)?;
        let layout = build_shard_layout(&subject, shard_path, parsed)?;
        stream_layout_payloads_into(&mut source, &layout, &mut snapshot)?;
        // `source` drops here: one open file per shard, never two.
    }
    scan_unreferenced_shards(&index, &open)?;
    Ok(snapshot)
}

/// Reads one single safetensors file into a snapshot under the given version
/// contract, streaming payload bytes one tensor at a time through one open
/// handle.
#[cfg(feature = "std")]
fn snapshot_from_single_file_via<S, F>(
    path: &std::path::Path,
    gate: VersionGate,
    open: F,
) -> Result<StateSnapshot>
where
    S: ByteSource,
    F: Fn(&std::path::Path) -> Result<S>,
{
    let mut source = open(path)?;
    let subject = format!("file `{}`", path.display());
    let parsed = read_safetensors_header(&mut source, &subject)?;
    if gate == VersionGate::StrictInc {
        require_state_format_version(&parsed.header)?;
    }
    let layout = build_shard_layout(&subject, path.to_path_buf(), parsed)?;
    let mut snapshot = StateSnapshot::new();
    stream_layout_payloads_into(&mut source, &layout, &mut snapshot)?;
    Ok(snapshot)
}

#[cfg(feature = "std")]
pub(crate) fn deserialize_snapshot_safetensors(
    path: &std::path::Path,
) -> anyhow::Result<StateSnapshot> {
    if looks_like_safetensors_index(path) {
        return deserialize_snapshot_safetensors_index(path);
    }
    Ok(snapshot_from_single_file_via(
        path,
        VersionGate::StrictInc,
        open_file_source,
    )?)
}

/// Parses a safetensors file into backend-neutral owned state, without
/// requiring an `incin.format.version` key.
///
/// [`deserialize_snapshot_safetensors`] is the loader behind `ModelExt::load`
/// and refuses an unversioned file, because that file was never written by a
/// versioned incin build and the version contract cannot say anything about
/// it. A file downloaded from an external source - the Hugging Face Hub,
/// most obviously - is unversioned for exactly that reason and is never
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
    Ok(snapshot_from_single_file_via(
        path,
        VersionGate::Foreign,
        open_file_source,
    )?)
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

/// A [`crate::nn::state::StateStream`] over safetensors files: the `path` is
/// either a sharded-checkpoint index or a single safetensors file, and every
/// payload reaches the caller one tensor at a time.
///
/// Construction validates everything that does *not* require payload bytes -
/// the index cross-checks, `total_size`, the unreferenced-sibling scan, and
/// every shard's header, roles, and dtypes - closing each file before opening
/// the next, so a checkpoint whose structure is wrong fails before the load
/// has staged anything. During placement at most one shard file is open: the
/// previous handle is dropped before the next is opened, and tensors are read
/// by byte range from the cached handle rather than by re-reading or
/// re-parsing the file. Single-file streams keep their one handle from
/// construction onward.
///
/// `F` is the opener; injecting it is how tests count opens and measure the
/// size of the largest single read without touching the filesystem.
#[cfg(feature = "std")]
pub(crate) struct SafetensorsStateStream<S, F> {
    /// One layout per shard (exactly one for a single-file stream), in sorted
    /// shard-name order for indexes.
    layouts: Vec<ShardLayout>,
    /// Tensor name to layout index; built only after cross-checks proved
    /// every name has exactly one owner.
    tensor_shard: BTreeMap<String, usize>,
    /// The shard currently open, if any: `(layout index, handle)`.
    active: Option<(usize, S)>,
    open: F,
}

#[cfg(feature = "std")]
impl<S, F> SafetensorsStateStream<S, F>
where
    S: ByteSource,
    F: Fn(&std::path::Path) -> Result<S>,
{
    /// Opens `path` - index or single file - through `open`, validating all
    /// header-level structure eagerly and leaving no shard open except, for a
    /// single file, its own handle.
    pub(crate) fn open_with(path: &std::path::Path, open: F) -> Result<Self> {
        if looks_like_safetensors_index(path) {
            Self::open_index(path, open)
        } else {
            Self::open_single(path, open)
        }
    }

    fn open_index(path: &std::path::Path, open: F) -> Result<Self> {
        use crate::nn::safetensors_index::SafetensorsIndex;

        let index = SafetensorsIndex::open(path)?;
        verify_total_size(&index)?;
        let mut layouts = Vec::new();
        for shard in index.shards() {
            let shard_path = index.shard_path(shard);
            let mut source = open(&shard_path)?;
            let subject = format!("shard `{shard}`");
            let parsed = read_safetensors_header(&mut source, &subject)?;
            cross_check_index_shard(&index, shard, &parsed)?;
            layouts.push(build_shard_layout(&subject, shard_path, parsed)?);
            // `source` drops here: construction holds at most one open file.
        }
        scan_unreferenced_shards(&index, &open)?;
        let mut tensor_shard = BTreeMap::new();
        for (layout_index, layout) in layouts.iter().enumerate() {
            for name in layout.tensors.keys() {
                let previous = tensor_shard.insert(name.clone(), layout_index);
                debug_assert!(
                    previous.is_none(),
                    "cross-checks proved every tensor has exactly one owning shard"
                );
            }
        }
        Ok(Self {
            layouts,
            tensor_shard,
            active: None,
            open,
        })
    }

    fn open_single(path: &std::path::Path, open: F) -> Result<Self> {
        let mut source = open(path)?;
        let subject = format!("file `{}`", path.display());
        let parsed = read_safetensors_header(&mut source, &subject)?;
        require_state_format_version(&parsed.header)?;
        let layout = build_shard_layout(&subject, path.to_path_buf(), parsed)?;
        let layouts = vec![layout];
        let mut tensor_shard = BTreeMap::new();
        for name in layouts[0].tensors.keys() {
            tensor_shard.insert(name.clone(), 0);
        }
        Ok(Self {
            layouts,
            tensor_shard,
            active: Some((0, source)),
            open,
        })
    }
}

#[cfg(feature = "std")]
impl<S, F> crate::nn::state::StateStream for SafetensorsStateStream<S, F>
where
    S: ByteSource,
    F: Fn(&std::path::Path) -> Result<S>,
{
    fn paths(&self) -> Result<alloc::collections::BTreeSet<StatePath>> {
        let mut paths = alloc::collections::BTreeSet::new();
        for layout in &self.layouts {
            for name in layout.tensors.keys() {
                paths.insert(StatePath::new(name)?);
            }
        }
        Ok(paths)
    }

    fn read(&mut self, path: &StatePath) -> Result<StateValue> {
        let name = path.as_str();
        let &layout_index =
            self.tensor_shard
                .get(name)
                .ok_or_else(|| Error::InvalidModuleState {
                    operation: "load state",
                    reason: ErrorMessage::new(format!("missing state path {path}")),
                })?;
        if self.active.as_ref().map(|(index, _)| *index) != Some(layout_index) {
            // Close the previous shard before opening the next: the number of
            // concurrently open files never exceeds one, including across the
            // switch.
            self.active = None;
            let shard_path = self.layouts[layout_index].path.clone();
            let source = (self.open)(&shard_path)?;
            self.active = Some((layout_index, source));
        }
        let (active, layouts) = (&mut self.active, &self.layouts);
        // The branch above leaves exactly this layout's handle in place (or
        // errors), and `tensor_shard` was built from these same layouts.
        let source = &mut active
            .as_mut()
            .expect("the active handle for this layout was ensured above")
            .1;
        let layout = &layouts[layout_index];
        let plan = layout
            .tensors
            .get(name)
            .expect("tensor_shard only points at names its layout contains");
        let bytes = read_tensor_bytes(source, layout, name, plan)?;
        build_state_value(name, &layout.subject, plan, bytes)
    }
}

/// The concrete stream [`SafetensorsStateStream::open`] returns: real files,
/// default opener.
#[cfg(feature = "std")]
type FileSafetensorsStream =
    SafetensorsStateStream<FileByteSource, fn(&std::path::Path) -> Result<FileByteSource>>;

#[cfg(feature = "std")]
impl SafetensorsStateStream<FileByteSource, fn(&std::path::Path) -> Result<FileByteSource>> {
    /// Opens `path` - a sharded index or a single safetensors file - with the
    /// default file opener. Single files must satisfy the strict
    /// [`ModelExt::load`] version contract; index files are external by
    /// construction and need no version key.
    pub(crate) fn open(path: &std::path::Path) -> Result<FileSafetensorsStream> {
        SafetensorsStateStream::open_with(
            path,
            open_file_source as fn(&std::path::Path) -> Result<FileByteSource>,
        )
    }
}

#[cfg(feature = "std")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Serialization formats modules understand.
pub enum Format {
    /// Hugging Face safetensors archive.
    Safetensors,
    /// Compact postcard binary (incin envelopes).
    Postcard,
    /// ONNX model export.
    ONNX,
}

#[cfg(feature = "std")]
/// Save/load conveniences implemented for modules.
pub trait ModelExt<B: Backend + crate::tensor::backend::VariableBackend> {
    /// Writes module state to `path` in `format`.
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
        match format {
            Format::Safetensors => {
                // Streamed placement: the file is validated header-first and
                // its tensors reach the module one at a time, so a failure
                // at any leaf clears staging and leaves the module exactly as
                // it was - the same failure-atomicity `load_state` provides
                // for snapshots, without ever holding the whole checkpoint on
                // the host.
                let mut stream: FileSafetensorsStream = SafetensorsStateStream::open(path)?;
                crate::nn::state::load_state_streaming::<B, _, _>(self, &mut stream)
            }
            Format::Postcard => {
                let snapshot =
                    deserialize_snapshot_postcard(path).map_err(|e| Error::Msg(e.to_string()))?;
                crate::nn::load_state::<B, _>(self, &snapshot)
            }
            Format::ONNX => Err(Error::Msg("ONNX is not a state format".into())),
        }
    }
}

#[cfg(all(test, feature = "std"))]
mod tests {
    use super::*;
    use crate::nn::state::StateStream;

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

    /// A [`ByteSource`] wrapper recording the size of every single read and
    /// the high-water mark of concurrently open sources: the measurement seam
    /// for bounded host residency (issue #13 acceptance 1). It wraps the real
    /// file opener, so the production loaders run unchanged underneath it.
    struct CountingSource {
        inner: FileByteSource,
        stats: std::sync::Arc<std::sync::Mutex<CountingStats>>,
    }

    #[derive(Default)]
    struct CountingStats {
        open: usize,
        concurrent: usize,
        max_concurrent: usize,
        max_read: usize,
    }

    impl Drop for CountingSource {
        fn drop(&mut self) {
            let mut guard = self.stats.lock().expect("stats mutex is not poisoned");
            guard.concurrent -= 1;
        }
    }

    impl ByteSource for CountingSource {
        fn len(&self) -> u64 {
            self.inner.len()
        }

        fn read_exact_at(&mut self, offset: u64, len: usize) -> std::io::Result<Vec<u8>> {
            {
                let mut guard = self.stats.lock().expect("stats mutex is not poisoned");
                guard.max_read = guard.max_read.max(len);
            }
            self.inner.read_exact_at(offset, len)
        }
    }

    /// A counting opener over the real file opener, sharing `stats` with the
    /// test that created it.
    fn counting_opener(
        stats: &std::sync::Arc<std::sync::Mutex<CountingStats>>,
    ) -> impl Fn(&std::path::Path) -> Result<CountingSource> {
        let stats = std::sync::Arc::clone(stats);
        move |path: &std::path::Path| -> Result<CountingSource> {
            let inner = FileByteSource::open(path)?;
            {
                let mut guard = stats.lock().expect("stats mutex is not poisoned");
                guard.open += 1;
                guard.concurrent += 1;
                guard.max_concurrent = guard.max_concurrent.max(guard.concurrent);
            }
            Ok(CountingSource {
                inner,
                stats: std::sync::Arc::clone(&stats),
            })
        }
    }

    /// Writes `snapshot` as two real shard files plus a
    /// `*.safetensors.index.json` declaring `total_size`, the layout
    /// `ModelExt::load` receives for a sharded checkpoint. Entries split by
    /// half of the sorted path list. Returns the index path and both shard
    /// paths in creation order (first half, second half).
    fn write_sharded_checkpoint(
        dir: &std::path::Path,
        snapshot: &StateSnapshot,
    ) -> (std::path::PathBuf, Vec<std::path::PathBuf>) {
        let first_name = "model-00001-of-00002.safetensors";
        let second_name = "model-00002-of-00002.safetensors";
        let midpoint = snapshot.len() / 2;
        let mut first = StateSnapshot::new();
        let mut second = StateSnapshot::new();
        for (index, (path, value)) in snapshot.iter().enumerate() {
            let target = if index < midpoint {
                &mut first
            } else {
                &mut second
            };
            target
                .insert(path.clone(), value.clone())
                .expect("snapshot paths are unique");
        }
        assert!(
            !first.is_empty() && !second.is_empty(),
            "both shards carry tensors"
        );
        let first_path = dir.join(first_name);
        let second_path = dir.join(second_name);
        serialize_snapshot_safetensors(&first, &first_path).expect("first shard is written");
        serialize_snapshot_safetensors(&second, &second_path).expect("second shard is written");
        let total = std::fs::metadata(&first_path)
            .expect("first shard stat")
            .len()
            + std::fs::metadata(&second_path)
                .expect("second shard stat")
                .len();
        let mut map = String::new();
        for (index, (path, _)) in snapshot.iter().enumerate() {
            let shard = if index < midpoint {
                first_name
            } else {
                second_name
            };
            if index > 0 {
                map.push(',');
            }
            map.push_str(&format!("\"{}\":\"{shard}\"", path.as_str()));
        }
        let json = format!("{{\"metadata\":{{\"total_size\":{total}}},\"weight_map\":{{{map}}}}}");
        let index_path = dir.join("model.safetensors.index.json");
        std::fs::write(&index_path, json).expect("index is written");
        (index_path, vec![first_path, second_path])
    }

    /// The header JSON length declared by a shard's 8-byte prefix.
    fn declared_header_len(path: &std::path::Path) -> usize {
        let bytes = std::fs::read(path).expect("shard is readable");
        u64::from_le_bytes(
            bytes[..SAFETENSORS_PREFIX_LEN]
                .try_into()
                .expect("a written shard has its 8-byte prefix"),
        ) as usize
    }

    /// The residency bound: no single read may exceed the header JSON or the
    /// largest tensor - the larger of the two, never the sum of the shards.
    fn residency_bound(snapshot: &StateSnapshot, shard_paths: &[std::path::PathBuf]) -> usize {
        snapshot
            .iter()
            .map(|(_, value)| value.bytes().len())
            .chain(shard_paths.iter().map(|path| declared_header_len(path)))
            .max()
            .expect("a checkpoint has at least one tensor or header")
    }

    /// Residency is measured at the reader seam (issue #13 acceptance 1): the
    /// counting [`ByteSource`] wraps the real file opener while the
    /// *production* sharded loader runs, recording the largest single
    /// `read_exact_at` length and the high-water mark of concurrently open
    /// sources. The assertions below are the acceptance criteria: one file
    /// open at a time, one open per shard (header and payloads share it), and
    /// no single read larger than the header or the largest tensor - which
    /// proves peak host residency is bounded by the larger of those two
    /// rather than by checkpoint size.
    #[test]
    fn a_sharded_snapshot_load_reads_one_tensor_range_at_a_time_with_one_file_open() {
        let dir = tempfile::tempdir().expect("temp dir");
        let expected = fixture();
        let (index_path, shard_paths) = write_sharded_checkpoint(dir.path(), &expected);
        let bound = residency_bound(&expected, &shard_paths);
        let total_bytes: usize = shard_paths
            .iter()
            .map(|path| std::fs::metadata(path).expect("shard stat").len() as usize)
            .sum();

        let stats = std::sync::Arc::new(std::sync::Mutex::new(CountingStats::default()));
        let actual =
            deserialize_snapshot_safetensors_index_via(&index_path, counting_opener(&stats))
                .expect("sharded fixture loads");
        assert_eq!(actual, expected);

        let stats = stats.lock().expect("stats mutex is not poisoned");
        assert_eq!(
            stats.max_concurrent, 1,
            "at most one shard file is ever open"
        );
        assert_eq!(
            stats.open, 2,
            "exactly one open per shard: the header and every payload share it"
        );
        assert!(
            stats.max_read <= bound,
            "no single read exceeds the header or the largest tensor: {} > {bound}",
            stats.max_read
        );
        assert!(
            stats.max_read < total_bytes,
            "the checkpoint is never read in one piece"
        );
    }

    /// The same residency contract through the placement machinery (issue
    /// #13 acceptance 1, sharded case): `StateStream::read` is exercised
    /// directly - construction validated every header first, then one tensor
    /// at a time is pulled while the counting opener watches, crossing shard
    /// switches. The restored-module equality guarantee is covered by the
    /// integration round-trip test; this test isolates the stream's I/O
    /// residency, which is identical for any backend.
    #[test]
    fn placement_streams_one_tensor_at_a_time_with_one_file_open() {
        let dir = tempfile::tempdir().expect("temp dir");
        let expected = fixture();
        let (index_path, shard_paths) = write_sharded_checkpoint(dir.path(), &expected);
        let bound = residency_bound(&expected, &shard_paths);
        let total_bytes: usize = shard_paths
            .iter()
            .map(|path| std::fs::metadata(path).expect("shard stat").len() as usize)
            .sum();

        let stats = std::sync::Arc::new(std::sync::Mutex::new(CountingStats::default()));
        let mut stream = SafetensorsStateStream::open_with(&index_path, counting_opener(&stats))
            .expect("stream validates every shard header");
        let paths: Vec<StatePath> = stream.paths().expect("paths list").into_iter().collect();
        assert_eq!(paths.len(), expected.len(), "every mapped path is readable");
        let mut biggest = 0usize;
        for path in &paths {
            let value = StateStream::read(&mut stream, path).expect("tensor reads");
            biggest = biggest.max(value.bytes().len());
        }
        let largest = expected
            .iter()
            .map(|(_, value)| value.bytes().len())
            .max()
            .expect("fixture has tensors");
        assert_eq!(biggest, largest, "largest tensor seen");

        let stats = stats.lock().expect("stats mutex is not poisoned");
        assert_eq!(
            stats.max_concurrent, 1,
            "at most one shard file is ever open, including across shard switches"
        );
        assert!(
            stats.open >= 2,
            "each shard was opened at least for its header: {}",
            stats.open
        );
        assert!(
            stats.max_read <= bound,
            "no single read exceeds the header or the largest tensor: {} > {bound}",
            stats.max_read
        );
        assert!(
            stats.max_read < total_bytes,
            "the checkpoint is never read in one piece"
        );
    }

    /// A read that fails partway through placement - here because the second
    /// shard file is deleted after the stream validated it but before any
    /// tensor from it is pulled - must surface a structured error naming the
    /// shard (issue #13 acceptance 2, I/O side). The surviving first shard's
    /// tensors read successfully first, so the failure lands mid-stream after
    /// successful reads. Module-atomicity on failure is covered by the
    /// integration shape-mismatch rollback test; this test isolates the
    /// stream-level error contract without needing a backend.
    #[test]
    fn deleting_a_shard_after_opening_the_stream_fails_the_load_without_touching_the_module() {
        let dir = tempfile::tempdir().expect("temp dir");
        let before = fixture();
        let (index_path, shard_paths) = write_sharded_checkpoint(dir.path(), &before);
        let mut stream = SafetensorsStateStream::open(&index_path)
            .expect("stream opens while both shards are present");
        std::fs::remove_file(&shard_paths[1]).expect("second shard deletes");

        let paths: Vec<StatePath> = stream.paths().expect("paths list").into_iter().collect();
        let mut succeeded = 0usize;
        let mut failure: Option<String> = None;
        for path in &paths {
            match StateStream::read(&mut stream, path) {
                Ok(_) => succeeded += 1,
                Err(error) => {
                    failure = Some(error.to_string());
                    break;
                }
            }
        }
        let message = failure.expect("the deleted shard must fail its read");
        assert!(
            message.contains("model-00002-of-00002.safetensors"),
            "the error names the shard it could not re-open, got: {message}"
        );
        assert!(
            succeeded >= 1,
            "the surviving shard's tensors were already read - failure landed mid-stream \
             after {succeeded} successful reads"
        );
    }

    // Compile-level proof (issue #13 acceptance 3) lives in
    // `tests/streaming_checkpoint.rs`, which coerces the public
    // `ModelExt::load` entry for `WgpuBackendImpl` to a function pointer -
    // instantiating `load_state_streaming::<WgpuBackendImpl, ..>` end to
    // end. It cannot live here: unit tests inside `src/` cannot link the
    // `incin_backends` dev-dependency (it cycles back into this crate's
    // rlib).
}
