//! `cargo incin doctor` — what this machine can actually run (`UX-014`).
//!
//! PROPOSALS.md sec. 2.3 specifies the report and closes with the two
//! properties that make it a support tool rather than a print statement: it is
//! read-only, and its output is stable human-readable text plus optional JSON.
//!
//! Both properties are structural here rather than aspirational.
//!
//! *Read-only* is why writeability is inferred from mode bits instead of by
//! attempting a write, and why the telemetry run directory is reported without
//! being resolved through `incin_telemetry::run_dir::default_run_dir`, which
//! creates it. A diagnostic that changes the thing it is diagnosing is not one.
//!
//! *Testable* is why every impure observation goes behind
//! [`Host`](crate::doctor::Host). Assembling
//! the report, deriving its findings, and rendering it are pure functions of
//! that trait's answers, so `tests/doctor.rs` can put a machine with three
//! GPUs and an unwritable cache in front of the doctor on a CI runner that has
//! neither. A report built from ambient hardware asserts nothing on the
//! machines this repository actually runs on.
//!
//! What is *not* covered is as deliberate. Metal is `MTL-001` and unbuilt, so
//! there is no Metal row: a row reporting "not available" for a backend that
//! does not exist reads as a hardware finding rather than an absent feature.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use incin_core::exec::{Capabilities, CapabilityQuery, LayoutClass, MathMode, SupportLevel};
use incin_core::shapes::OperationKind;
use incin_core::tensor::device::{DeviceId, DeviceKind};
use incin_core::tensor::dtype::DTypeId;
use serde::Serialize;

/// Version of the JSON document this module emits.
///
/// A support workflow that parses the report needs to know when the shape it
/// parses has changed, and the crate version does not tell it — most releases
/// will not touch this schema. Bumped on any change to a key or to the meaning
/// of a value; a new *finding code* is not a schema change, since consumers
/// already have to tolerate codes they do not know.
pub const SCHEMA_VERSION: u32 = 1;

/// The backend families the report covers, in the order it prints them.
const DEVICE_ORDER: &[DeviceKind] = &[DeviceKind::Cpu, DeviceKind::Cuda, DeviceKind::Wgpu];

// ============================================================================
// The observations — one trait for the whole impure surface
// ============================================================================

/// Everything the doctor asks the machine.
///
/// This is the entire boundary between the report and the world. Every method
/// answers a question whose answer differs between machines; nothing else in
/// this module reads the environment, runs a process, or touches the
/// filesystem. [`HostMachine`] answers them for real, and a test answers them
/// however it needs to.
pub trait Host {
    /// The `incin` version this build came from.
    ///
    /// On the trait rather than an `env!` so a golden-text test can pin a
    /// version and keep passing across releases.
    fn incin_version(&self) -> String;

    /// `rustc --version`, or `None` when it could not be run.
    fn rustc_version(&self) -> Option<String>;

    /// Every Cargo feature the report knows about, and whether it is on.
    fn features(&self) -> Vec<Feature>;

    /// The instruction-set extensions the CPU kernels branch on.
    ///
    /// Only extensions with an actual branch behind them appear. Listing what
    /// the machine supports but no kernel uses would suggest the report says
    /// something about performance that it does not.
    fn cpu_isa(&self) -> Vec<IsaFeature>;

    /// Whether this build contains the family at all — a `cfg!`, not a probe.
    fn compiled_in(&self, kind: DeviceKind) -> bool;

    /// Whether hardware for the family answers right now.
    fn probe(&self, kind: DeviceKind) -> Option<DeviceId>;

    /// The on-disk caches the runtime reads or writes.
    fn caches(&self) -> Vec<Cache>;
}

// ============================================================================
// The report
// ============================================================================

/// A Cargo feature and whether this build enabled it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Feature {
    /// Cargo feature name.
    pub name: String,
    /// Whether the feature is enabled in this build.
    pub enabled: bool,
}

impl Feature {
    /// Creates a feature observation for a report.
    #[must_use]
    pub fn new(name: &str, enabled: bool) -> Self {
        Self {
            name: name.to_string(),
            enabled,
        }
    }
}

/// An instruction-set extension a CPU kernel branches on.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IsaFeature {
    /// CPU extension name.
    pub name: String,
    /// Whether the host reports support for the extension.
    pub available: bool,
}

impl IsaFeature {
    /// Creates an instruction-set observation for a report.
    #[must_use]
    pub fn new(name: &str, available: bool) -> Self {
        Self {
            name: name.to_string(),
            available,
        }
    }
}

/// What the doctor could determine about a cache directory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum CacheState {
    /// The directory exists and its mode bits permit writing.
    Writable,
    /// The directory exists and its mode bits do not permit writing.
    NotWritable,
    /// The path is known but nothing is there yet.
    Absent,
    /// The path is configuration-dependent and no configuration was found.
    Unset,
    /// The feature that owns this cache is not in this build.
    NotCompiled,
}

impl CacheState {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Writable => "writable",
            Self::NotWritable => "not writable",
            Self::Absent => "absent",
            Self::Unset => "unset",
            Self::NotCompiled => "not compiled",
        }
    }
}

/// One cache directory the runtime uses.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Cache {
    /// Stable name of the cache.
    pub name: String,
    /// Configured cache path, when one was discoverable.
    pub path: Option<String>,
    /// Observed state of the cache path.
    pub state: CacheState,
    /// Why the state is what it is, when the state alone does not say.
    pub detail: Option<String>,
}

impl Cache {
    /// Creates a cache observation from an optional path and state.
    #[must_use]
    pub fn new(name: &str, path: Option<PathBuf>, state: CacheState) -> Self {
        Self {
            name: name.to_string(),
            path: path.map(|p| p.display().to_string()),
            state,
            detail: None,
        }
    }

    /// Adds a human-readable explanation to the observation.
    #[must_use]
    pub fn with_detail(mut self, detail: &str) -> Self {
        self.detail = Some(detail.to_string());
        self
    }
}

/// One backend family: whether it is in the build, and whether it answered.
///
/// These are different questions with different remedies — a missing feature
/// is a rebuild, a missing device is a driver — which is why they are separate
/// fields rather than one tri-state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceReport {
    /// Device family name.
    pub kind: String,
    /// Whether support for the family was compiled into this build.
    pub compiled_in: bool,
    /// Whether a device of this family answered the probe.
    pub available: bool,
    /// Device ordinal when one was reported.
    pub ordinal: Option<usize>,
}

/// One representative operation, asked of one available device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Probe {
    /// Device name used for the probe.
    pub device: String,
    /// Operation family that was queried.
    pub operation: String,
    /// Dtype used for the query.
    pub dtype: String,
    /// Rank used for the query.
    pub rank: usize,
    /// Whether the query exercised training support.
    pub training: bool,
    /// Registry support result rendered for the report.
    pub support: String,
    /// The registry's reason, when the answer was `unsupported`.
    pub reason: Option<String>,
}

/// How much a finding matters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    /// Worth knowing; nothing is broken.
    Note,
    /// This build will not do something a reader may expect it to.
    Warning,
    /// This build cannot run.
    Error,
}

impl Severity {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Note => "note",
            Self::Warning => "warning",
            Self::Error => "error",
        }
    }
}

/// Something the report concluded from what it observed.
///
/// The `code` is the stable part. A support workflow greps for the code and a
/// human reads the message, so the message is free to improve while the code
/// is not free to change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Finding {
    /// Importance of the finding.
    pub severity: Severity,
    /// Stable machine-readable finding code.
    pub code: String,
    /// Human-readable explanation.
    pub message: String,
    /// Optional remediation guidance.
    pub remedy: Option<String>,
}

impl Finding {
    fn new(severity: Severity, code: &str, message: String, remedy: Option<&str>) -> Self {
        Self {
            severity,
            code: code.to_string(),
            message,
            remedy: remedy.map(ToString::to_string),
        }
    }
}

/// Versions of the two things a bug report is always asked for first.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Toolchain {
    /// Incin version that produced the report.
    pub incin: String,
    /// Rust compiler version, when it could be queried.
    pub rustc: Option<String>,
}

/// The complete diagnostic report.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Report {
    /// Version of the report schema.
    pub schema_version: u32,
    /// Toolchain versions that produced the report.
    pub toolchain: Toolchain,
    /// Cargo features observed in the build.
    pub features: Vec<Feature>,
    /// CPU extensions used by kernel dispatch.
    pub cpu_isa: Vec<IsaFeature>,
    /// Backend availability observations.
    pub devices: Vec<DeviceReport>,
    /// Runtime cache observations.
    pub caches: Vec<Cache>,
    /// Representative operation probes.
    pub probes: Vec<Probe>,
    /// Findings derived from the observations.
    pub findings: Vec<Finding>,
}

/// One representative operation the report asks every available device about.
struct ProbeSpec {
    operation: OperationKind,
    dtype: DTypeId,
    rank: usize,
    training: bool,
}

const fn spec(operation: OperationKind, dtype: DTypeId, rank: usize, training: bool) -> ProbeSpec {
    ProbeSpec {
        operation,
        dtype,
        rank,
        training,
    }
}

/// The operations the report probes, chosen to be representative rather than
/// exhaustive.
///
/// One per coarse family a user actually reaches for, plus two deliberate
/// negatives: `f64` reduction and `f16` matmul are unsupported on the CPU
/// registry today, so the probe section demonstrates that it reports both
/// answers rather than only confirming what works.
const PROBES: &[ProbeSpec] = &[
    spec(OperationKind::Pointwise, DTypeId::F32, 2, false),
    spec(OperationKind::Pointwise, DTypeId::F32, 2, true),
    spec(OperationKind::MatMul, DTypeId::F32, 2, false),
    spec(OperationKind::MatMul, DTypeId::F16, 2, false),
    spec(OperationKind::Reduction, DTypeId::F32, 2, false),
    spec(OperationKind::Reduction, DTypeId::F64, 2, false),
    spec(OperationKind::Conv2d, DTypeId::F32, 4, true),
    spec(OperationKind::Normalization, DTypeId::F32, 2, true),
];

impl Report {
    /// Assemble the report from a [`Host`].
    ///
    /// Pure given the host: the same answers produce the same report, which is
    /// what makes a golden test of the rendered output meaningful.
    #[must_use]
    pub fn gather(host: &dyn Host) -> Self {
        let toolchain = Toolchain {
            incin: host.incin_version(),
            rustc: host.rustc_version(),
        };
        let features = host.features();
        let cpu_isa = host.cpu_isa();

        // Each family is probed exactly once and the answer is reused. Probing
        // is not a cheap pure query — `detect::probe` builds a WGPU instance
        // and enumerates adapters, and retains a CUDA primary context — so
        // asking twice is asking the driver to do it all again. The first
        // draft did ask twice, once here and once to decide what to probe
        // capabilities for, and under `cargo test --workspace` (which unifies
        // the `wgpu` feature in) two tests doing that concurrently took the
        // process down with a SIGSEGV inside the adapter enumeration.
        let detected: Vec<Option<DeviceId>> =
            DEVICE_ORDER.iter().map(|&kind| host.probe(kind)).collect();

        let devices: Vec<DeviceReport> = DEVICE_ORDER
            .iter()
            .zip(&detected)
            .map(|(&kind, device)| DeviceReport {
                kind: device_name(kind).to_string(),
                compiled_in: host.compiled_in(kind),
                available: device.is_some(),
                ordinal: device.map(DeviceId::ordinal),
            })
            .collect();

        let caches = host.caches();

        // Only devices that answered. A capability answer for hardware that is
        // not there is a claim the machine cannot honour, and the "compiled in
        // but unavailable" case is already reported one section up.
        let probes: Vec<Probe> = DEVICE_ORDER
            .iter()
            .zip(&detected)
            .filter(|(_, device)| device.is_some())
            .flat_map(|(&kind, _)| PROBES.iter().map(move |probe| run_probe(kind, probe)))
            .collect();

        let findings = findings(&toolchain, &features, &cpu_isa, &devices, &caches);

        Self {
            schema_version: SCHEMA_VERSION,
            toolchain,
            features,
            cpu_isa,
            devices,
            caches,
            probes,
            findings,
        }
    }

    /// [`EXIT_FINDINGS`] when anything is a [`Severity::Error`], else
    /// [`EXIT_OK`].
    ///
    /// Warnings do not fail: a build with CUDA compiled in and no GPU attached
    /// is a normal laptop, and a doctor that exits non-zero on it is a doctor
    /// people stop running.
    #[must_use]
    pub fn exit_code(&self) -> i32 {
        if self
            .findings
            .iter()
            .any(|finding| finding.severity == Severity::Error)
        {
            EXIT_FINDINGS
        } else {
            EXIT_OK
        }
    }

    /// The stable human-readable rendering.
    ///
    /// Every line is `key: value` under a bracketed section header, in a fixed
    /// section order, with no column padding. Padding is what makes a text
    /// report unstable — a single long path shifts every other line — and the
    /// format is the part sec. 2.3 asks to be stable.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut out = String::new();

        out.push_str("[toolchain]\n");
        let _ = writeln!(out, "incin: {}", self.toolchain.incin);
        let _ = writeln!(
            out,
            "rustc: {}",
            self.toolchain.rustc.as_deref().unwrap_or("unknown")
        );

        out.push_str("\n[features]\n");
        for feature in &self.features {
            let _ = writeln!(
                out,
                "{}: {}",
                feature.name,
                if feature.enabled { "on" } else { "off" }
            );
        }

        out.push_str("\n[cpu]\n");
        if self.cpu_isa.is_empty() {
            out.push_str("isa: none used by any kernel on this target\n");
        } else {
            for isa in &self.cpu_isa {
                let _ = writeln!(
                    out,
                    "{}: {}",
                    isa.name,
                    if isa.available {
                        "available"
                    } else {
                        "unavailable (scalar path)"
                    }
                );
            }
        }

        out.push_str("\n[devices]\n");
        for device in &self.devices {
            let state = match (device.compiled_in, device.ordinal) {
                (true, Some(ordinal)) => {
                    format!("compiled, available as {}:{ordinal}", device.kind)
                }
                (true, None) => "compiled, no device found".to_string(),
                // A device that answers without being compiled in cannot
                // happen on a real host, but a mocked one can say it, and
                // silently printing "not compiled" would hide the
                // contradiction rather than show it.
                (false, Some(ordinal)) => {
                    format!("not compiled, but answered as {}:{ordinal}", device.kind)
                }
                (false, None) => "not compiled".to_string(),
            };
            let _ = writeln!(out, "{}: {state}", device.kind);
        }

        out.push_str("\n[caches]\n");
        for cache in &self.caches {
            let mut line = match &cache.path {
                Some(path) => format!("{path} ({})", cache.state.as_str()),
                None => cache.state.as_str().to_string(),
            };
            if let Some(detail) = &cache.detail {
                line.push_str(&format!(" — {detail}"));
            }
            let _ = writeln!(out, "{}: {line}", cache.name);
        }

        out.push_str("\n[probes]\n");
        if self.probes.is_empty() {
            out.push_str("none: no device answered\n");
        }
        for probe in &self.probes {
            let mode = if probe.training {
                "training"
            } else {
                "inference"
            };
            let answer = match &probe.reason {
                Some(reason) => format!("{} ({reason})", probe.support),
                None => probe.support.clone(),
            };
            let _ = writeln!(
                out,
                "{} {} {} rank {} {mode}: {answer}",
                probe.device, probe.operation, probe.dtype, probe.rank
            );
        }

        out.push_str("\n[findings]\n");
        if self.findings.is_empty() {
            out.push_str("none\n");
        }
        for finding in &self.findings {
            let _ = writeln!(
                out,
                "{} {}: {}",
                finding.severity.as_str(),
                finding.code,
                finding.message
            );
            if let Some(remedy) = &finding.remedy {
                let _ = writeln!(out, "  remedy: {remedy}");
            }
        }

        out
    }

    /// The JSON rendering, for CI and support reports.
    ///
    /// # Errors
    ///
    /// Only if the report cannot be serialized, which cannot happen for the
    /// types above; the signature keeps the `unwrap` out of the library.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

fn run_probe(kind: DeviceKind, spec: &ProbeSpec) -> Probe {
    let query = CapabilityQuery {
        operation: incin_core::exec::OperationIdentity::Builtin(spec.operation),
        dtype: spec.dtype.descriptor(),
        layout: LayoutClass::Contiguous,
        rank: spec.rank,
        training: spec.training,
        math_mode: MathMode::Precise,
    };
    let level = incin_backends::capability::registry(kind).support(&query);
    let (support, reason) = match level {
        SupportLevel::Native => ("native", None),
        SupportLevel::Composed => ("composed", None),
        SupportLevel::Fallback => ("fallback", None),
        SupportLevel::Unsupported(why) => ("unsupported", Some(why.to_string())),
    };
    Probe {
        device: device_name(kind).to_string(),
        operation: spec.operation.name().to_string(),
        dtype: dtype_name(spec.dtype).to_string(),
        rank: spec.rank,
        training: spec.training,
        support: support.to_string(),
        reason,
    }
}

/// Derive the findings from everything observed.
///
/// Pure, and the only place a conclusion is drawn: the sections above report,
/// this decides. Emitted errors first, then warnings, then notes, and within
/// each in the fixed order of the inputs, so the list is deterministic.
///
/// A rejected probe is deliberately *not* a finding. The first draft made one,
/// and the healthy-machine test showed what that means in practice: an
/// ordinary CPU laptop opened its report with two notes saying `f16` matmul
/// and `f64` reduction are unsupported, which is not a fault, is not
/// actionable, and is already in the probe section a few lines up. A section
/// that always has something in it is a section people stop reading.
fn findings(
    toolchain: &Toolchain,
    _features: &[Feature],
    cpu_isa: &[IsaFeature],
    devices: &[DeviceReport],
    caches: &[Cache],
) -> Vec<Finding> {
    let mut out = Vec::new();

    if devices.iter().all(|device| !device.compiled_in) {
        out.push(Finding::new(
            Severity::Error,
            "no-backend-compiled",
            "no backend family is compiled into this build, so nothing can execute".to_string(),
            Some("rebuild with at least one of the cpu, cuda, or wgpu features"),
        ));
    }

    if toolchain.rustc.is_none() {
        out.push(Finding::new(
            Severity::Warning,
            "toolchain-unknown",
            "rustc could not be run, so the toolchain version is unknown".to_string(),
            Some("check that rustc is on PATH"),
        ));
    }

    // The warning sec. 2.3 names explicitly: "warnings when a feature is
    // compiled but its runtime is unavailable".
    for device in devices {
        if device.compiled_in && !device.available {
            out.push(Finding::new(
                Severity::Warning,
                "backend-unavailable",
                format!(
                    "{} is compiled into this build but no device answered",
                    device.kind
                ),
                Some(match device.kind.as_str() {
                    "cuda" => "check that a driver is installed and visible to this process",
                    "wgpu" => "check that a non-software adapter is visible to this process",
                    _ => "check that the backend's runtime is present",
                }),
            ));
        }
    }

    for cache in caches {
        if cache.state == CacheState::NotWritable {
            out.push(Finding::new(
                Severity::Warning,
                "cache-not-writable",
                format!(
                    "the {} cache at {} is not writable",
                    cache.name,
                    cache.path.as_deref().unwrap_or("an unknown path")
                ),
                Some("fix the directory's permissions or point the cache elsewhere"),
            ));
        }
    }

    for isa in cpu_isa {
        if !isa.available {
            out.push(Finding::new(
                Severity::Note,
                "isa-unavailable",
                format!(
                    "{} is not available, so the CPU kernels that branch on it take their scalar path",
                    isa.name
                ),
                None,
            ));
        }
    }

    out
}

// `device_name` and `dtype_name` were private copies here until `UX-013`. Both
// needed a `_ => "unknown"` arm, because these enums are `#[non_exhaustive]`
// outside `incin-core` — which meant a dtype added later would have rendered as
// the literal string "unknown" in a support report rather than failing to
// build. `DeviceKind::name` and `DTypeId::name` live beside the enums now, where
// the match is exhaustive, and the generated capability tables read the same
// two functions. §2.10 asks for one manifest behind the matrix, the reference
// docs and these probes; two spellings of `f32` would be two manifests.
fn device_name(kind: DeviceKind) -> &'static str {
    kind.name()
}

fn dtype_name(dtype: DTypeId) -> &'static str {
    dtype.name()
}

// ============================================================================
// The real machine
// ============================================================================

/// [`Host`] answered by the machine this process is running on.
#[derive(Debug, Clone, Copy, Default)]
pub struct HostMachine;

/// Every Cargo feature of the `incin` crate, in the order the report prints
/// them, paired with whether this build has it.
///
/// Written out rather than derived, because there is no way to enumerate one's
/// own features at runtime — which means this list can fall behind
/// `Cargo.toml`. `tests/doctor.rs` reads the manifest and fails when it does.
fn compiled_features() -> Vec<Feature> {
    vec![
        Feature::new("std", cfg!(feature = "std")),
        Feature::new("nightly", cfg!(feature = "nightly")),
        Feature::new("cpu", cfg!(feature = "cpu")),
        Feature::new("cpu-blas", cfg!(feature = "cpu-blas")),
        Feature::new("cuda", cfg!(feature = "cuda")),
        Feature::new("wgpu", cfg!(feature = "wgpu")),
        Feature::new("external-candle", cfg!(feature = "external-candle")),
        Feature::new("metal", cfg!(feature = "metal")),
        Feature::new("metal-mps", cfg!(feature = "metal-mps")),
        Feature::new("autotune", cfg!(feature = "autotune")),
        Feature::new("train", cfg!(feature = "train")),
        Feature::new("distributed", cfg!(feature = "distributed")),
        Feature::new(
            "distributed-reference",
            cfg!(feature = "distributed-reference"),
        ),
        Feature::new("distributed-nccl", cfg!(feature = "distributed-nccl")),
        Feature::new("telemetry", cfg!(feature = "telemetry")),
        Feature::new("test-utils", cfg!(feature = "test-utils")),
        Feature::new("backend-authoring", cfg!(feature = "backend-authoring")),
        Feature::new("compiled", cfg!(feature = "compiled")),
        Feature::new("hardware-tests", cfg!(feature = "hardware-tests")),
    ]
}

/// The instruction-set extensions the CPU kernels actually branch on.
///
/// `x86_64` is the only target where the branch is a runtime one:
/// `cpu/ops/elementwise_kernel.rs` calls `is_x86_feature_detected!("avx2")` and
/// falls back to a scalar loop. On `aarch64` the NEON path is selected by
/// `cfg`, so it is present whenever the build targets it; on `wasm32` the
/// `simd128` path is a `target_feature` decided at compile time. Elsewhere no
/// kernel branches at all, and the report says so rather than inventing a row.
fn detected_isa() -> Vec<IsaFeature> {
    #[cfg(target_arch = "x86_64")]
    {
        vec![IsaFeature::new(
            "avx2",
            std::arch::is_x86_feature_detected!("avx2"),
        )]
    }
    #[cfg(target_arch = "aarch64")]
    {
        vec![IsaFeature::new("neon", true)]
    }
    #[cfg(target_arch = "wasm32")]
    {
        vec![IsaFeature::new("simd128", cfg!(target_feature = "simd128"))]
    }
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "wasm32"
    )))]
    {
        Vec::new()
    }
}

/// Classify a cache directory without touching it.
///
/// The permission bit is read from the directory's metadata rather than probed
/// by creating a file, because the doctor is read-only by contract. That is a
/// weaker answer than a write attempt — it does not account for ACLs, for
/// read-only mounts, or for a full disk — and the report says "not writable"
/// only when the mode bits are unambiguous about it.
///
/// Public so the read-only contract can be asserted directly: pointing this at
/// a path that does not exist must leave it not existing.
#[must_use]
pub fn cache_state(path: &Path) -> CacheState {
    match std::fs::metadata(path) {
        Ok(metadata) if metadata.permissions().readonly() => CacheState::NotWritable,
        Ok(_) => CacheState::Writable,
        Err(_) => CacheState::Absent,
    }
}

/// The telemetry run directory, resolved without creating it.
#[cfg(feature = "telemetry")]
fn telemetry_cache() -> Cache {
    match incin_telemetry::run_dir::default_run_dir_path() {
        Ok(path) => {
            let state = cache_state(&path);
            Cache::new("telemetry-runs", Some(path), state)
        }
        Err(_) => Cache::new("telemetry-runs", None, CacheState::Unset)
            .with_detail("no XDG data directory could be resolved"),
    }
}

#[cfg(not(feature = "telemetry"))]
fn telemetry_cache() -> Cache {
    Cache::new("telemetry-runs", None, CacheState::NotCompiled)
        .with_detail("enable the telemetry feature")
}

impl Host for HostMachine {
    fn incin_version(&self) -> String {
        env!("CARGO_PKG_VERSION").to_string()
    }

    fn rustc_version(&self) -> Option<String> {
        let output = std::process::Command::new("rustc")
            .arg("--version")
            .output()
            .ok()?;
        if !output.status.success() {
            return None;
        }
        let text = String::from_utf8(output.stdout).ok()?;
        let trimmed = text.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_string())
    }

    fn features(&self) -> Vec<Feature> {
        compiled_features()
    }

    fn cpu_isa(&self) -> Vec<IsaFeature> {
        detected_isa()
    }

    fn compiled_in(&self, kind: DeviceKind) -> bool {
        incin_backends::detect::is_compiled_in(kind)
    }

    fn probe(&self, kind: DeviceKind) -> Option<DeviceId> {
        incin_backends::detect::probe(kind)
    }

    fn caches(&self) -> Vec<Cache> {
        let hub = match std::env::var_os("INCIN_HUB_CACHE_DIR") {
            Some(dir) => {
                let path = PathBuf::from(dir);
                let state = cache_state(&path);
                Cache::new("hub", Some(path), state)
            }
            // `incin-data` only overrides `hf-hub`'s own cache location when
            // this variable is set, and where `hf-hub` puts it otherwise is
            // that crate's business. Reporting a path this module guessed
            // would be worse than reporting none.
            None => Cache::new("hub", None, CacheState::Unset)
                .with_detail("INCIN_HUB_CACHE_DIR is unset; hf-hub chooses"),
        };
        vec![hub, telemetry_cache()]
    }
}

// ============================================================================
// The subcommand
// ============================================================================

/// Exit code for a report that found nothing wrong.
pub const EXIT_OK: i32 = 0;
/// Exit code for a report containing a [`Severity::Error`] finding.
pub const EXIT_FINDINGS: i32 = 1;
/// Exit code for a malformed invocation, distinct from a bad report.
///
/// Separate from [`EXIT_FINDINGS`] because a CI job that runs the doctor as a
/// gate needs to tell "this machine cannot run incin" from "you typed the flag
/// wrong", and both are non-zero.
pub const EXIT_USAGE: i32 = 2;

/// Run `cargo incin doctor`, returning what to print and what to exit with.
///
/// Returns the output rather than printing it so the whole subcommand,
/// argument parsing included, is testable.
///
/// The argument vocabulary is closed. `CI-005` found `#[module]` accepting
/// anything that contained the right substring, which turned a typo into a
/// silent behaviour change; an unrecognized flag here is an error naming the
/// ones that exist.
#[must_use]
pub fn run(args: &[String]) -> (String, i32) {
    let mut json = false;
    for arg in args {
        match arg.as_str() {
            "--json" => json = true,
            other => {
                return (
                    format!("error: unknown argument `{other}` for doctor (expected --json)\n"),
                    EXIT_USAGE,
                );
            }
        }
    }

    let report = Report::gather(&HostMachine);
    let rendered = if json {
        match report.to_json() {
            Ok(text) => format!("{text}\n"),
            Err(error) => {
                return (
                    format!("error: could not render JSON: {error}\n"),
                    EXIT_USAGE,
                );
            }
        }
    } else {
        report.to_text()
    };
    (rendered, report.exit_code())
}
