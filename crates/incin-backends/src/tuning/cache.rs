//! Atomic, checksummed, and bounded persistent tuning records.
//!
//! The cache is deliberately candidate-agnostic. A record stores an opaque
//! winner plus the digest of the legal candidate set which produced it.
//! Callers must compare that digest and decode/revalidate the winner before
//! use. Imported cache data is therefore a hint, never a proof.

use super::identity::{
    BackendIdentity, IdentityError, StaticBackend, TuningEnvironmentFingerprint,
};
use alloc::{
    collections::BTreeMap,
    string::{String, ToString},
    vec::Vec,
};
use core::{
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    marker::PhantomData,
    time::Duration,
};
use incin_core::shapes::dynamic::Dyn;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    ffi::OsString,
    fs::{self, File, OpenOptions},
    io::{self, Write as _},
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering as AtomicOrdering},
    time::{SystemTime, UNIX_EPOCH},
};

/// Persistent format version.
pub const CACHE_SCHEMA_VERSION: u32 = 1;
const MAX_NAMESPACE_BYTES: usize = 64;
const MAX_PROBLEM_BYTES: usize = 1024;
const MAX_WINNER_BYTES: usize = 4096;

/// Why a persistent-cache operation failed.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CacheError {
    /// A configured bound cannot describe any cache.
    #[error("invalid tuning-cache limits: {message}")]
    InvalidLimits {
        /// Human-readable constraint failure.
        message: String,
    },
    /// A key or record field was invalid.
    #[error("invalid tuning-cache record: {message}")]
    InvalidRecord {
        /// Human-readable constraint failure.
        message: String,
    },
    /// The record cannot fit even after every older entry is evicted.
    #[error("tuning-cache record exceeds the configured {maximum_bytes}-byte file bound")]
    RecordTooLarge {
        /// Maximum serialized cache size.
        maximum_bytes: usize,
    },
    /// A filesystem operation failed.
    #[error("failed to {operation} tuning cache `{path}`: {source}")]
    Io {
        /// Operation being attempted.
        operation: &'static str,
        /// Affected path.
        path: PathBuf,
        /// Underlying error.
        #[source]
        source: io::Error,
    },
    /// JSON serialization failed.
    #[error("failed to serialize tuning cache: {message}")]
    Serialization {
        /// Serializer error.
        message: String,
    },
}

fn io_error(operation: &'static str, path: &Path, source: io::Error) -> CacheError {
    CacheError::Io {
        operation,
        path: path.to_path_buf(),
        source,
    }
}

/// Hard persistent-cache limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CacheLimits {
    max_entries: usize,
    max_bytes: usize,
    max_age: Option<Duration>,
}

impl CacheLimits {
    /// Constructs nonzero entry and byte bounds.
    pub fn new(
        max_entries: usize,
        max_bytes: usize,
        max_age: Option<Duration>,
    ) -> core::result::Result<Self, CacheError> {
        if max_entries == 0 {
            return Err(CacheError::InvalidLimits {
                message: "max_entries must be nonzero".to_string(),
            });
        }
        if max_bytes == 0 {
            return Err(CacheError::InvalidLimits {
                message: "max_bytes must be nonzero".to_string(),
            });
        }
        let minimum_bytes = serialized(&BTreeMap::new())?.len();
        if max_bytes < minimum_bytes {
            return Err(CacheError::InvalidLimits {
                message: format!(
                    "max_bytes {max_bytes} cannot hold the {minimum_bytes}-byte empty cache"
                ),
            });
        }
        Ok(Self {
            max_entries,
            max_bytes,
            max_age,
        })
    }

    /// Maximum number of records.
    #[must_use]
    pub const fn max_entries(self) -> usize {
        self.max_entries
    }

    /// Maximum serialized file length.
    #[must_use]
    pub const fn max_bytes(self) -> usize {
        self.max_bytes
    }

    /// Maximum record age, if configured.
    #[must_use]
    pub const fn max_age(self) -> Option<Duration> {
        self.max_age
    }
}

impl Default for CacheLimits {
    fn default() -> Self {
        Self {
            max_entries: 1024,
            max_bytes: 4 * 1024 * 1024,
            max_age: Some(Duration::from_secs(90 * 24 * 60 * 60)),
        }
    }
}

/// A stable key for one tuning problem.
///
/// Static backends construct `CacheKey<D>` from a matching typed environment.
/// `CacheKey<Dyn>` stores the runtime backend and checks it when projected
/// back to a static marker.
pub struct CacheKey<D = Dyn> {
    namespace: String,
    backend: BackendIdentity,
    environment_digest: u64,
    problem: String,
    marker: PhantomData<fn() -> D>,
}

impl<D: StaticBackend> CacheKey<D> {
    /// Constructs a key whose backend is known at compile time.
    pub fn new(
        namespace: &str,
        environment: &TuningEnvironmentFingerprint<D>,
        problem: &str,
    ) -> core::result::Result<Self, CacheError> {
        Self::from_parts(
            namespace,
            environment.device().backend(),
            environment.digest(),
            problem,
        )
    }

    /// Erases the static backend marker.
    #[must_use]
    pub fn erase(self) -> CacheKey<Dyn> {
        CacheKey {
            namespace: self.namespace,
            backend: self.backend,
            environment_digest: self.environment_digest,
            problem: self.problem,
            marker: PhantomData,
        }
    }
}

impl CacheKey<Dyn> {
    /// Constructs a key for a runtime-selected, already-validated environment.
    pub fn new_dyn(
        namespace: &str,
        environment: &TuningEnvironmentFingerprint<Dyn>,
        problem: &str,
    ) -> core::result::Result<Self, CacheError> {
        Self::from_parts(
            namespace,
            environment.device().backend(),
            environment.digest(),
            problem,
        )
    }

    /// Projects a runtime key to a statically known backend.
    pub fn try_into_static<D: StaticBackend>(
        self,
    ) -> core::result::Result<CacheKey<D>, IdentityError> {
        if self.backend != D::BACKEND {
            return Err(IdentityError::BackendMismatch {
                expected: D::BACKEND.name(),
                actual: self.backend.name(),
            });
        }
        Ok(CacheKey {
            namespace: self.namespace,
            backend: self.backend,
            environment_digest: self.environment_digest,
            problem: self.problem,
            marker: PhantomData,
        })
    }
}

impl<D> CacheKey<D> {
    fn from_parts(
        namespace: &str,
        backend: BackendIdentity,
        environment_digest: u64,
        problem: &str,
    ) -> core::result::Result<Self, CacheError> {
        validate_text("namespace", namespace, MAX_NAMESPACE_BYTES)?;
        validate_text("problem", problem, MAX_PROBLEM_BYTES)?;
        Ok(Self {
            namespace: namespace.to_string(),
            backend,
            environment_digest,
            problem: problem.to_string(),
            marker: PhantomData,
        })
    }

    /// Cache namespace, such as `kernel` or `collective`.
    #[must_use]
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// Backend family.
    #[must_use]
    pub const fn backend(&self) -> BackendIdentity {
        self.backend
    }

    /// Stable digest of the device/compiler environment.
    #[must_use]
    pub const fn environment_digest(&self) -> u64 {
        self.environment_digest
    }

    /// Canonical problem identity.
    #[must_use]
    pub fn problem(&self) -> &str {
        &self.problem
    }

    /// Stable, length-delimited key digest.
    #[must_use]
    pub fn digest(&self) -> u64 {
        Digest::new()
            .field(b"incin.tuning.cache-key.v1")
            .text(&self.namespace)
            .text(self.backend.name())
            .number(self.environment_digest)
            .text(&self.problem)
            .finish()
    }
}

impl<D> Clone for CacheKey<D> {
    fn clone(&self) -> Self {
        Self {
            namespace: self.namespace.clone(),
            backend: self.backend,
            environment_digest: self.environment_digest,
            problem: self.problem.clone(),
            marker: PhantomData,
        }
    }
}

impl<D> fmt::Debug for CacheKey<D> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CacheKey")
            .field("namespace", &self.namespace)
            .field("backend", &self.backend)
            .field("environment_digest", &self.environment_digest)
            .field("problem", &self.problem)
            .finish()
    }
}

impl<D> PartialEq for CacheKey<D> {
    fn eq(&self, other: &Self) -> bool {
        self.namespace == other.namespace
            && self.backend == other.backend
            && self.environment_digest == other.environment_digest
            && self.problem == other.problem
    }
}

impl<D> Eq for CacheKey<D> {}

impl<D> PartialOrd for CacheKey<D> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<D> Ord for CacheKey<D> {
    fn cmp(&self, other: &Self) -> Ordering {
        (
            &self.namespace,
            self.backend,
            self.environment_digest,
            &self.problem,
        )
            .cmp(&(
                &other.namespace,
                other.backend,
                other.environment_digest,
                &other.problem,
            ))
    }
}

impl<D> Hash for CacheKey<D> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.namespace.hash(state);
        self.backend.hash(state);
        self.environment_digest.hash(state);
        self.problem.hash(state);
    }
}

#[derive(Serialize, Deserialize)]
struct WireCacheKey {
    namespace: String,
    backend: BackendIdentity,
    environment_digest: u64,
    problem: String,
}

impl Serialize for CacheKey<Dyn> {
    fn serialize<S: Serializer>(&self, serializer: S) -> core::result::Result<S::Ok, S::Error> {
        WireCacheKey {
            namespace: self.namespace.clone(),
            backend: self.backend,
            environment_digest: self.environment_digest,
            problem: self.problem.clone(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for CacheKey<Dyn> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> core::result::Result<Self, D::Error> {
        use serde::de::Error as _;

        let wire = WireCacheKey::deserialize(deserializer)?;
        Self::from_parts(
            &wire.namespace,
            wire.backend,
            wire.environment_digest,
            &wire.problem,
        )
        .map_err(D::Error::custom)
    }
}

/// How a persistent winner was selected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
#[non_exhaustive]
pub enum MeasurementMethod {
    /// Synchronized device-event measurements.
    Measured,
    /// Deterministic analytical or rule-based choice.
    Heuristic,
    /// An explicitly imported offline profile.
    ProfileGuided,
    /// A coordinated multi-rank warmup.
    CoordinatedWarmup,
}

/// One persistent tuning result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CacheRecord {
    key: CacheKey<Dyn>,
    created_unix_ms: u64,
    method: MeasurementMethod,
    sample_count: u32,
    median_ns: Option<u64>,
    legal_candidates_digest: u64,
    winner: String,
}

#[derive(Deserialize)]
struct WireCacheRecord {
    key: CacheKey<Dyn>,
    created_unix_ms: u64,
    method: MeasurementMethod,
    sample_count: u32,
    median_ns: Option<u64>,
    legal_candidates_digest: u64,
    winner: String,
}

impl<'de> Deserialize<'de> for CacheRecord {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> core::result::Result<Self, D::Error> {
        use serde::de::Error as _;

        let wire = WireCacheRecord::deserialize(deserializer)?;
        Self::new_at(
            wire.key,
            wire.created_unix_ms,
            wire.method,
            wire.sample_count,
            wire.median_ns,
            wire.legal_candidates_digest,
            &wire.winner,
        )
        .map_err(D::Error::custom)
    }
}

impl CacheRecord {
    /// Constructs a record at the current wall-clock time.
    pub fn new(
        key: CacheKey<Dyn>,
        method: MeasurementMethod,
        sample_count: u32,
        median_ns: Option<u64>,
        legal_candidates_digest: u64,
        winner: &str,
    ) -> core::result::Result<Self, CacheError> {
        Self::new_at(
            key,
            unix_time_ms()?,
            method,
            sample_count,
            median_ns,
            legal_candidates_digest,
            winner,
        )
    }

    /// Constructs a record with an explicit timestamp for deterministic
    /// profile import and testing.
    pub fn new_at(
        key: CacheKey<Dyn>,
        created_unix_ms: u64,
        method: MeasurementMethod,
        sample_count: u32,
        median_ns: Option<u64>,
        legal_candidates_digest: u64,
        winner: &str,
    ) -> core::result::Result<Self, CacheError> {
        validate_text("winner", winner, MAX_WINNER_BYTES)?;
        if matches!(
            method,
            MeasurementMethod::Measured | MeasurementMethod::CoordinatedWarmup
        ) && (sample_count == 0 || median_ns.is_none())
        {
            return Err(CacheError::InvalidRecord {
                message: "measured records require a nonzero sample count and median".to_string(),
            });
        }
        let record = Self {
            key,
            created_unix_ms,
            method,
            sample_count,
            median_ns,
            legal_candidates_digest,
            winner: winner.to_string(),
        };
        record.validate()?;
        Ok(record)
    }

    fn validate(&self) -> core::result::Result<(), CacheError> {
        validate_text("namespace", &self.key.namespace, MAX_NAMESPACE_BYTES)?;
        validate_text("problem", &self.key.problem, MAX_PROBLEM_BYTES)?;
        validate_text("winner", &self.winner, MAX_WINNER_BYTES)?;
        if matches!(
            self.method,
            MeasurementMethod::Measured | MeasurementMethod::CoordinatedWarmup
        ) && (self.sample_count == 0 || self.median_ns.is_none())
        {
            return Err(CacheError::InvalidRecord {
                message: "measured records require a nonzero sample count and median".to_string(),
            });
        }
        Ok(())
    }

    /// Record key.
    #[must_use]
    pub const fn key(&self) -> &CacheKey<Dyn> {
        &self.key
    }

    /// Creation time in milliseconds since the Unix epoch.
    #[must_use]
    pub const fn created_unix_ms(&self) -> u64 {
        self.created_unix_ms
    }

    /// Selection method.
    #[must_use]
    pub const fn method(&self) -> MeasurementMethod {
        self.method
    }

    /// Number of synchronized samples.
    #[must_use]
    pub const fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Median duration when the method measured one.
    #[must_use]
    pub const fn median_ns(&self) -> Option<u64> {
        self.median_ns
    }

    /// Digest of the legal candidate set at commit time.
    #[must_use]
    pub const fn legal_candidates_digest(&self) -> u64 {
        self.legal_candidates_digest
    }

    /// Opaque winner encoding. The caller must parse and revalidate it.
    #[must_use]
    pub fn winner(&self) -> &str {
        &self.winner
    }
}

/// Recovery performed while opening or merging a cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CacheRecovery {
    quarantined_path: PathBuf,
    reason: String,
}

impl CacheRecovery {
    /// Path holding the rejected bytes.
    #[must_use]
    pub fn quarantined_path(&self) -> &Path {
        &self.quarantined_path
    }

    /// Why the file was rejected.
    #[must_use]
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

/// A process-local view backed by an atomically replaced JSON file.
pub struct PersistentTuningCache {
    path: PathBuf,
    lock_path: PathBuf,
    limits: CacheLimits,
    entries: BTreeMap<CacheKey<Dyn>, CacheRecord>,
    recovery: Option<CacheRecovery>,
}

impl fmt::Debug for PersistentTuningCache {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PersistentTuningCache")
            .field("path", &self.path)
            .field("limits", &self.limits)
            .field("entries", &self.entries.len())
            .field("recovery", &self.recovery)
            .finish()
    }
}

impl PersistentTuningCache {
    /// Opens a cache, quarantining malformed, mismatched-schema, invalid, or
    /// checksum-failing bytes rather than trusting them.
    pub fn open(
        path: impl AsRef<Path>,
        limits: CacheLimits,
    ) -> core::result::Result<Self, CacheError> {
        Self::open_at(path, limits, unix_time_ms()?)
    }

    /// Opens a cache at an explicit time for deterministic profile tools and
    /// tests.
    pub fn open_at(
        path: impl AsRef<Path>,
        limits: CacheLimits,
        now_unix_ms: u64,
    ) -> core::result::Result<Self, CacheError> {
        let path = path.as_ref().to_path_buf();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)
                .map_err(|source| io_error("create directory for", &path, source))?;
        }
        let lock_path = sidecar_path(&path, ".lock");
        let lock = open_lock(&lock_path)?;
        fs2::FileExt::lock_exclusive(&lock)
            .map_err(|source| io_error("lock", &lock_path, source))?;
        let loaded = load_entries(&path, limits, now_unix_ms)?;
        if loaded.changed {
            write_entries(&path, &loaded.entries)?;
        }
        fs2::FileExt::unlock(&lock).map_err(|source| io_error("unlock", &lock_path, source))?;
        Ok(Self {
            path,
            lock_path,
            limits,
            entries: loaded.entries,
            recovery: loaded.recovery,
        })
    }

    /// Cache path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Recovery performed during the latest disk merge.
    #[must_use]
    pub const fn recovery(&self) -> Option<&CacheRecovery> {
        self.recovery.as_ref()
    }

    /// Number of loaded records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no records are loaded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Looks up a record only when the current legal candidate set matches.
    ///
    /// The returned winner remains opaque and untrusted; the caller must parse
    /// it and prove it belongs to the current legal set.
    #[must_use]
    pub fn lookup(
        &self,
        key: &CacheKey<Dyn>,
        legal_candidates_digest: u64,
    ) -> Option<&CacheRecord> {
        self.entries
            .get(key)
            .filter(|record| record.legal_candidates_digest == legal_candidates_digest)
    }

    /// Merges a record with any writes made by other processes, prunes the
    /// configured bounds, and atomically replaces the cache file.
    pub fn commit(&mut self, record: CacheRecord) -> core::result::Result<(), CacheError> {
        self.commit_at(record, unix_time_ms()?)
    }

    /// Commits at an explicit time for deterministic profile tools and tests.
    pub fn commit_at(
        &mut self,
        record: CacheRecord,
        now_unix_ms: u64,
    ) -> core::result::Result<(), CacheError> {
        record.validate()?;
        let committed_key = record.key.clone();
        let lock = open_lock(&self.lock_path)?;
        fs2::FileExt::lock_exclusive(&lock)
            .map_err(|source| io_error("lock", &self.lock_path, source))?;
        let mut loaded = load_entries(&self.path, self.limits, now_unix_ms)?;
        loaded.entries.insert(committed_key.clone(), record);
        prune_age(&mut loaded.entries, self.limits, now_unix_ms);
        prune_count(&mut loaded.entries, self.limits.max_entries);
        prune_size(&mut loaded.entries, self.limits.max_bytes)?;
        if !loaded.entries.contains_key(&committed_key) {
            fs2::FileExt::unlock(&lock)
                .map_err(|source| io_error("unlock", &self.lock_path, source))?;
            return Err(CacheError::RecordTooLarge {
                maximum_bytes: self.limits.max_bytes,
            });
        }
        write_entries(&self.path, &loaded.entries)?;
        fs2::FileExt::unlock(&lock)
            .map_err(|source| io_error("unlock", &self.lock_path, source))?;
        self.entries = loaded.entries;
        if loaded.recovery.is_some() {
            self.recovery = loaded.recovery;
        }
        Ok(())
    }

    /// Removes all persistent records through the same atomic write path.
    pub fn clear(&mut self) -> core::result::Result<(), CacheError> {
        let lock = open_lock(&self.lock_path)?;
        fs2::FileExt::lock_exclusive(&lock)
            .map_err(|source| io_error("lock", &self.lock_path, source))?;
        let entries = BTreeMap::new();
        write_entries(&self.path, &entries)?;
        fs2::FileExt::unlock(&lock)
            .map_err(|source| io_error("unlock", &self.lock_path, source))?;
        self.entries = entries;
        Ok(())
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct CacheEnvelope {
    schema: u32,
    entries: Vec<CacheRecord>,
    checksum: String,
}

#[derive(Serialize)]
struct ChecksummedBody<'a> {
    schema: u32,
    entries: &'a [CacheRecord],
}

struct LoadResult {
    entries: BTreeMap<CacheKey<Dyn>, CacheRecord>,
    recovery: Option<CacheRecovery>,
    changed: bool,
}

fn load_entries(
    path: &Path,
    limits: CacheLimits,
    now_unix_ms: u64,
) -> core::result::Result<LoadResult, CacheError> {
    if let Ok(metadata) = fs::metadata(path)
        && metadata.len() as usize > limits.max_bytes
    {
        return quarantined_empty(
            path,
            format!(
                "file size {} exceeds maximum allowed cache bytes {}",
                metadata.len(),
                limits.max_bytes
            ),
        );
    }
    let bytes = match fs::read(path) {
        Ok(bytes) => bytes,
        Err(source) if source.kind() == io::ErrorKind::NotFound => {
            return Ok(LoadResult {
                entries: BTreeMap::new(),
                recovery: None,
                changed: false,
            });
        }
        Err(source) => return Err(io_error("read", path, source)),
    };
    let envelope: CacheEnvelope = match serde_json::from_slice(&bytes) {
        Ok(envelope) => envelope,
        Err(error) => {
            return quarantined_empty(path, format!("invalid JSON: {error}"));
        }
    };
    if envelope.schema != CACHE_SCHEMA_VERSION {
        return quarantined_empty(
            path,
            format!(
                "schema {} does not match {}",
                envelope.schema, CACHE_SCHEMA_VERSION
            ),
        );
    }
    let expected = checksum(&envelope.entries)?;
    if envelope.checksum != expected {
        return quarantined_empty(
            path,
            format!("checksum {} does not match {expected}", envelope.checksum),
        );
    }
    let mut entries = BTreeMap::new();
    for record in envelope.entries {
        if let Err(error) = record.validate() {
            return quarantined_empty(path, error.to_string());
        }
        if entries.insert(record.key.clone(), record).is_some() {
            return quarantined_empty(path, "duplicate cache key".to_string());
        }
    }
    let original_len = entries.len();
    prune_age(&mut entries, limits, now_unix_ms);
    prune_count(&mut entries, limits.max_entries);
    prune_size(&mut entries, limits.max_bytes)?;
    Ok(LoadResult {
        changed: entries.len() != original_len || bytes.len() > limits.max_bytes,
        entries,
        recovery: None,
    })
}

fn quarantined_empty(path: &Path, reason: String) -> core::result::Result<LoadResult, CacheError> {
    let quarantined_path = quarantine_path(path);
    fs::rename(path, &quarantined_path).map_err(|source| io_error("quarantine", path, source))?;
    Ok(LoadResult {
        entries: BTreeMap::new(),
        recovery: Some(CacheRecovery {
            quarantined_path,
            reason,
        }),
        changed: false,
    })
}

fn prune_age(
    entries: &mut BTreeMap<CacheKey<Dyn>, CacheRecord>,
    limits: CacheLimits,
    now_unix_ms: u64,
) {
    let Some(max_age) = limits.max_age else {
        return;
    };
    let maximum_ms = max_age.as_millis().min(u128::from(u64::MAX)) as u64;
    entries.retain(|_, record| now_unix_ms.saturating_sub(record.created_unix_ms) <= maximum_ms);
}

fn prune_count(entries: &mut BTreeMap<CacheKey<Dyn>, CacheRecord>, maximum: usize) {
    while entries.len() > maximum {
        evict_oldest(entries);
    }
}

fn prune_size(
    entries: &mut BTreeMap<CacheKey<Dyn>, CacheRecord>,
    maximum_bytes: usize,
) -> core::result::Result<(), CacheError> {
    while serialized(entries)?.len() > maximum_bytes && !entries.is_empty() {
        evict_oldest(entries);
    }
    Ok(())
}

fn evict_oldest(entries: &mut BTreeMap<CacheKey<Dyn>, CacheRecord>) {
    let oldest = entries
        .iter()
        .min_by(|(left_key, left), (right_key, right)| {
            (left.created_unix_ms, *left_key).cmp(&(right.created_unix_ms, *right_key))
        })
        .map(|(key, _)| key.clone());
    if let Some(oldest) = oldest {
        entries.remove(&oldest);
    }
}

fn checksum(entries: &[CacheRecord]) -> core::result::Result<String, CacheError> {
    let body = serde_json::to_vec(&ChecksummedBody {
        schema: CACHE_SCHEMA_VERSION,
        entries,
    })
    .map_err(|error| CacheError::Serialization {
        message: error.to_string(),
    })?;
    Ok(format!("{:016x}", Digest::new().bytes(&body).finish()))
}

fn serialized(
    entries: &BTreeMap<CacheKey<Dyn>, CacheRecord>,
) -> core::result::Result<Vec<u8>, CacheError> {
    let records: Vec<_> = entries.values().cloned().collect();
    let envelope = CacheEnvelope {
        schema: CACHE_SCHEMA_VERSION,
        checksum: checksum(&records)?,
        entries: records,
    };
    serde_json::to_vec_pretty(&envelope).map_err(|error| CacheError::Serialization {
        message: error.to_string(),
    })
}

fn write_entries(
    path: &Path,
    entries: &BTreeMap<CacheKey<Dyn>, CacheRecord>,
) -> core::result::Result<(), CacheError> {
    let bytes = serialized(entries)?;
    let temporary = temporary_path(path);
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&temporary)
        .map_err(|source| io_error("create temporary file for", path, source))?;
    let write_result = (|| {
        file.write_all(&bytes)
            .map_err(|source| io_error("write temporary file for", path, source))?;
        file.sync_all()
            .map_err(|source| io_error("sync temporary file for", path, source))?;
        fs::rename(&temporary, path)
            .map_err(|source| io_error("atomically replace", path, source))?;
        sync_parent(path)?;
        Ok(())
    })();
    if write_result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    write_result
}

fn open_lock(path: &Path) -> core::result::Result<File, CacheError> {
    OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(path)
        .map_err(|source| io_error("open lock for", path, source))
}

#[cfg(unix)]
fn sync_parent(path: &Path) -> core::result::Result<(), CacheError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|source| io_error("sync parent directory for", path, source))
}

#[cfg(not(unix))]
fn sync_parent(_: &Path) -> core::result::Result<(), CacheError> {
    Ok(())
}

fn validate_text(
    field: &'static str,
    value: &str,
    maximum: usize,
) -> core::result::Result<(), CacheError> {
    if value.is_empty() {
        return Err(CacheError::InvalidRecord {
            message: format!("{field} must not be empty"),
        });
    }
    if value.len() > maximum {
        return Err(CacheError::InvalidRecord {
            message: format!("{field} is {} bytes; maximum is {maximum}", value.len()),
        });
    }
    if value.trim() != value || value.chars().any(char::is_control) {
        return Err(CacheError::InvalidRecord {
            message: format!("{field} is not canonical text"),
        });
    }
    Ok(())
}

fn unix_time_ms() -> core::result::Result<u64, CacheError> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| CacheError::InvalidRecord {
            message: format!("system clock precedes Unix epoch: {error}"),
        })?;
    u64::try_from(duration.as_millis()).map_err(|_| CacheError::InvalidRecord {
        message: "system clock exceeds u64 milliseconds".to_string(),
    })
}

fn sidecar_path(path: &Path, suffix: &str) -> PathBuf {
    let mut name = OsString::from(path.as_os_str());
    name.push(suffix);
    PathBuf::from(name)
}

fn quarantine_path(path: &Path) -> PathBuf {
    static NONCE: AtomicU64 = AtomicU64::new(0);
    let nonce = NONCE.fetch_add(1, AtomicOrdering::Relaxed);
    sidecar_path(
        path,
        &format!(
            ".corrupt-{}-{}-{nonce}",
            std::process::id(),
            unix_time_ms().unwrap_or(0)
        ),
    )
}

fn temporary_path(path: &Path) -> PathBuf {
    static NONCE: AtomicU64 = AtomicU64::new(0);
    let nonce = NONCE.fetch_add(1, AtomicOrdering::Relaxed);
    sidecar_path(path, &format!(".tmp-{}-{nonce}", std::process::id()))
}

#[derive(Clone, Copy)]
struct Digest(u64);

impl Digest {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;

    const fn new() -> Self {
        Self(Self::OFFSET)
    }

    fn bytes(mut self, bytes: &[u8]) -> Self {
        for &byte in bytes {
            self.0 = (self.0 ^ u64::from(byte)).wrapping_mul(Self::PRIME);
        }
        self
    }

    fn field(self, bytes: &[u8]) -> Self {
        self.number(bytes.len() as u64).bytes(bytes)
    }

    fn text(self, value: &str) -> Self {
        self.field(value.as_bytes())
    }

    fn number(self, value: u64) -> Self {
        self.bytes(&value.to_le_bytes())
    }

    const fn finish(self) -> u64 {
        self.0
    }
}
