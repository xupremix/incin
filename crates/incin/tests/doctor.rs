#![cfg(feature = "std")]

//! `UX-014`: `cargo incin doctor` reports what this machine can run.
//!
//! Almost every test here runs against a machine that does not exist. That is
//! the point rather than a shortcut: the report's whole job is to describe
//! hardware, and a test that describes *this* runner's hardware asserts one
//! configuration — the one with no GPU, which is every runner in this
//! repository's CI. Behind `Host`, a three-GPU machine with an unwritable
//! cache and a missing toolchain costs a struct literal.
//!
//! Two tests do run against the real machine. One is a smoke test that the
//! whole path works end to end; the other is the guard that catches the
//! failure mode a mocked suite invites, which is the mock drifting away from
//! the thing it stands in for.

use std::path::{Path, PathBuf};

use incin::doctor::{
    Cache, CacheState, EXIT_FINDINGS, EXIT_OK, EXIT_USAGE, Feature, Host, HostMachine, IsaFeature,
    Report, SCHEMA_VERSION, Severity, cache_state, run,
};
use incin::prelude::{DeviceId, DeviceKind};

// ============================================================================
// The machine that does not exist
// ============================================================================

#[derive(Debug, Clone)]
struct FakeHost {
    incin: String,
    rustc: Option<String>,
    features: Vec<Feature>,
    isa: Vec<IsaFeature>,
    compiled: Vec<DeviceKind>,
    present: Vec<DeviceId>,
    caches: Vec<Cache>,
}

impl FakeHost {
    /// A plain CPU laptop with nothing wrong with it.
    ///
    /// The baseline every other case is a single edit away from, so a test
    /// that provokes one finding provokes exactly that one and the assertion
    /// "this produced no other findings" means something.
    fn healthy() -> Self {
        Self {
            incin: "9.9.9".to_string(),
            rustc: Some("rustc 1.99.0 (deadbeef1 2030-01-01)".to_string()),
            features: vec![Feature::new("std", true), Feature::new("cpu", true)],
            isa: vec![IsaFeature::new("avx2", true)],
            compiled: vec![DeviceKind::Cpu],
            present: vec![DeviceId::cpu()],
            caches: vec![Cache::new(
                "hub",
                Some(PathBuf::from("/var/cache/incin/hub")),
                CacheState::Writable,
            )],
        }
    }
}

impl Host for FakeHost {
    fn incin_version(&self) -> String {
        self.incin.clone()
    }
    fn rustc_version(&self) -> Option<String> {
        self.rustc.clone()
    }
    fn features(&self) -> Vec<Feature> {
        self.features.clone()
    }
    fn cpu_isa(&self) -> Vec<IsaFeature> {
        self.isa.clone()
    }
    fn compiled_in(&self, kind: DeviceKind) -> bool {
        self.compiled.contains(&kind)
    }
    fn probe(&self, kind: DeviceKind) -> Option<DeviceId> {
        self.present
            .iter()
            .find(|device| device.kind() == kind)
            .copied()
    }
    fn caches(&self) -> Vec<Cache> {
        self.caches.clone()
    }
}

fn codes(report: &Report) -> Vec<&str> {
    report
        .findings
        .iter()
        .map(|finding| finding.code.as_str())
        .collect()
}

// ============================================================================
// The stable text
// ============================================================================

/// The rendering `UX-014` calls "stable human-readable text", pinned whole.
///
/// A golden test is the only thing that makes "stable" mean anything: without
/// it, the format is stable until someone changes it, which is not a property.
/// It also pins the probe list, so adding or dropping a representative
/// operation is a decision rather than a diff nobody sees.
///
/// The machine is deliberately unlike this one — three backends compiled, two
/// present, an unwritable cache, no AVX2, and the deprecated feature alias on.
///
/// The capability answers are not mocked. `Host` fakes which hardware is
/// *there*; the registries in `incin-backends` are static data compiled into
/// every build, so a mocked CUDA device gets CUDA's real registrations. Every
/// probe line below was checked against `crates/incin-backends/src/capability/tables.rs`
/// rather than recorded from output — the first draft had `matmul f16` and
/// `reduction f64` the wrong way round on CUDA, which is exactly the mistake a
/// recorded golden would have preserved.
#[test]
fn the_text_rendering_is_stable() {
    let host = FakeHost {
        features: vec![
            Feature::new("std", true),
            Feature::new("cpu", true),
            Feature::new("cuda", true),
            Feature::new("wgpu", true),
        ],
        isa: vec![IsaFeature::new("avx2", false)],
        compiled: vec![DeviceKind::Cpu, DeviceKind::Cuda, DeviceKind::Wgpu],
        present: vec![DeviceId::cpu(), DeviceId::cuda(2)],
        caches: vec![
            Cache::new(
                "hub",
                Some(PathBuf::from("/var/cache/incin/hub")),
                CacheState::NotWritable,
            ),
            Cache::new("telemetry-runs", None, CacheState::Unset)
                .with_detail("no XDG data directory could be resolved"),
        ],
        ..FakeHost::healthy()
    };

    let expected = "\
[toolchain]
incin: 9.9.9
rustc: rustc 1.99.0 (deadbeef1 2030-01-01)

[features]
std: on
cpu: on
cuda: on
wgpu: on

[cpu]
avx2: unavailable (scalar path)

[devices]
cpu: compiled, available as cpu:0
cuda: compiled, available as cuda:2
wgpu: compiled, no device found

[caches]
hub: /var/cache/incin/hub (not writable)
telemetry-runs: unset — no XDG data directory could be resolved

[probes]
cpu pointwise f32 rank 2 inference: native
cpu pointwise f32 rank 2 training: native
cpu matmul f32 rank 2 inference: native
cpu matmul f16 rank 2 inference: unsupported (dtype f16 is unsupported for matmul)
cpu reduction f32 rank 2 inference: native
cpu reduction f64 rank 2 inference: unsupported (dtype f64 is unsupported for reduction)
cpu conv2d f32 rank 4 training: native
cpu normalization f32 rank 2 training: native
cuda pointwise f32 rank 2 inference: native
cuda pointwise f32 rank 2 training: native
cuda matmul f32 rank 2 inference: native
cuda matmul f16 rank 2 inference: unsupported (dtype f16 is unsupported for matmul)
cuda reduction f32 rank 2 inference: native
cuda reduction f64 rank 2 inference: native
cuda conv2d f32 rank 4 training: native
cuda normalization f32 rank 2 training: unsupported (operation normalization is not registered)

[findings]
warning backend-unavailable: wgpu is compiled into this build but no device answered
  remedy: check that a non-software adapter is visible to this process
warning cache-not-writable: the hub cache at /var/cache/incin/hub is not writable
  remedy: fix the directory's permissions or point the cache elsewhere
note isa-unavailable: avx2 is not available, so the CPU kernels that branch on it take their scalar path
";

    assert_eq!(Report::gather(&host).to_text(), expected);
}

/// Findings are ordered errors, then warnings, then notes.
///
/// A support report is read top-down and often only the top is read, so the
/// thing that stops the build has to be above the thing that is merely worth
/// knowing.
#[test]
fn findings_are_ordered_by_severity() {
    let host = FakeHost {
        rustc: None,
        isa: vec![IsaFeature::new("avx2", false)],
        compiled: Vec::new(),
        present: Vec::new(),
        ..FakeHost::healthy()
    };
    let report = Report::gather(&host);
    let severities: Vec<Severity> = report
        .findings
        .iter()
        .map(|finding| finding.severity)
        .collect();

    assert_eq!(
        severities,
        vec![Severity::Error, Severity::Warning, Severity::Note],
        "{:?}",
        codes(&report)
    );
}

// ============================================================================
// The JSON
// ============================================================================

#[test]
fn the_json_carries_its_schema_version_and_a_fixed_key_set() {
    let report = Report::gather(&FakeHost::healthy());
    let json: serde_json::Value =
        serde_json::from_str(&report.to_json().expect("the report serializes")).unwrap();

    assert_eq!(json["schema_version"], SCHEMA_VERSION);

    let mut keys: Vec<&str> = json
        .as_object()
        .expect("the report is a JSON object")
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "caches",
            "cpu_isa",
            "devices",
            "features",
            "findings",
            "probes",
            "schema_version",
            "toolchain",
        ],
        "the JSON key set changed; that is a schema change, so bump SCHEMA_VERSION"
    );
}

/// The JSON says the same thing the text says.
///
/// Two renderers is two chances to disagree, and the one nobody reads by eye
/// is the one that drifts.
#[test]
fn the_json_and_the_text_agree_about_what_was_found() {
    let host = FakeHost {
        compiled: vec![DeviceKind::Cpu, DeviceKind::Cuda],
        ..FakeHost::healthy()
    };
    let report = Report::gather(&host);
    let json: serde_json::Value =
        serde_json::from_str(&report.to_json().expect("the report serializes")).unwrap();
    let text = report.to_text();

    let findings = json["findings"].as_array().expect("findings is an array");
    assert_eq!(findings.len(), 1);
    for finding in findings {
        let code = finding["code"].as_str().unwrap();
        let message = finding["message"].as_str().unwrap();
        assert!(text.contains(code), "{code} missing from the text");
        assert!(text.contains(message), "{message:?} missing from the text");
    }
}

#[test]
fn the_json_reports_every_probe_the_text_does() {
    let report = Report::gather(&FakeHost::healthy());
    let json: serde_json::Value =
        serde_json::from_str(&report.to_json().expect("the report serializes")).unwrap();

    let probes = json["probes"].as_array().expect("probes is an array");
    assert_eq!(probes.len(), report.probes.len());
    assert!(!probes.is_empty(), "a CPU host must probe something");
    for probe in probes {
        assert!(
            !probe["support"].as_str().unwrap().is_empty(),
            "a probe with no answer is not a probe"
        );
    }
}

// ============================================================================
// One test per finding, each provoked by a machine that produces it
// ============================================================================

#[test]
fn a_healthy_machine_produces_no_findings_and_exits_zero() {
    let report = Report::gather(&FakeHost::healthy());
    assert_eq!(codes(&report), Vec::<&str>::new());
    assert_eq!(report.exit_code(), EXIT_OK);
}

/// The warning sec. 2.3 names in as many words: "warnings when a feature is
/// compiled but its runtime is unavailable".
#[test]
fn a_compiled_backend_with_no_hardware_is_a_warning() {
    let host = FakeHost {
        compiled: vec![DeviceKind::Cpu, DeviceKind::Cuda, DeviceKind::Wgpu],
        present: vec![DeviceId::cpu()],
        ..FakeHost::healthy()
    };
    let report = Report::gather(&host);

    assert_eq!(
        codes(&report),
        ["backend-unavailable", "backend-unavailable"]
    );
    assert!(
        report
            .findings
            .iter()
            .all(|f| f.severity == Severity::Warning)
    );
    assert!(report.findings[0].message.contains("cuda"));
    assert!(report.findings[1].message.contains("wgpu"));
    // The remedies differ, because the thing to go check differs: a driver for
    // CUDA, an adapter for WGPU. A shared remedy string would be the tell that
    // the finding is generic where it should not be.
    assert_ne!(report.findings[0].remedy, report.findings[1].remedy);
    // It is a warning, so a laptop with the cuda feature on still exits zero.
    assert_eq!(report.exit_code(), EXIT_OK);
}

/// The one case that is an error rather than a warning.
#[test]
fn a_build_with_no_backend_at_all_is_an_error() {
    let host = FakeHost {
        compiled: Vec::new(),
        present: Vec::new(),
        ..FakeHost::healthy()
    };
    let report = Report::gather(&host);

    assert_eq!(codes(&report), ["no-backend-compiled"]);
    assert_eq!(report.findings[0].severity, Severity::Error);
    assert_eq!(report.exit_code(), EXIT_FINDINGS);
}

/// A device that is compiled in *and* available is not a finding.
///
/// The negative half of `a_compiled_backend_with_no_hardware_is_a_warning`: a
/// check that fires on the healthy case is worse than no check, because the
/// warning people learn to ignore is the one that is always there.
#[test]
fn a_backend_that_is_both_compiled_and_present_says_nothing() {
    let host = FakeHost {
        compiled: vec![DeviceKind::Cpu, DeviceKind::Cuda],
        present: vec![DeviceId::cpu(), DeviceId::cuda(0)],
        ..FakeHost::healthy()
    };
    assert_eq!(codes(&Report::gather(&host)), Vec::<&str>::new());
}

#[test]
fn a_missing_toolchain_version_is_a_warning() {
    let host = FakeHost {
        rustc: None,
        ..FakeHost::healthy()
    };
    let report = Report::gather(&host);

    assert_eq!(codes(&report), ["toolchain-unknown"]);
    assert!(report.to_text().contains("rustc: unknown"));
}

#[test]
fn an_unwritable_cache_is_a_warning_naming_its_path() {
    let host = FakeHost {
        caches: vec![Cache::new(
            "hub",
            Some(PathBuf::from("/read/only/hub")),
            CacheState::NotWritable,
        )],
        ..FakeHost::healthy()
    };
    let report = Report::gather(&host);

    assert_eq!(codes(&report), ["cache-not-writable"]);
    assert!(report.findings[0].message.contains("/read/only/hub"));
}

/// A cache that is merely absent is not a problem.
///
/// It is the normal state before the first run, and warning about it would
/// mean every fresh install starts with a warning.
#[test]
fn an_absent_or_unset_cache_is_not_a_finding() {
    for state in [
        CacheState::Absent,
        CacheState::Unset,
        CacheState::NotCompiled,
    ] {
        let host = FakeHost {
            caches: vec![Cache::new("hub", Some(PathBuf::from("/nowhere")), state)],
            ..FakeHost::healthy()
        };
        assert_eq!(
            codes(&Report::gather(&host)),
            Vec::<&str>::new(),
            "{state:?}"
        );
    }
}

#[test]
fn an_absent_isa_extension_is_a_note_about_the_scalar_path() {
    let host = FakeHost {
        isa: vec![IsaFeature::new("avx2", false)],
        ..FakeHost::healthy()
    };
    let report = Report::gather(&host);

    assert_eq!(codes(&report), ["isa-unavailable"]);
    assert_eq!(report.findings[0].severity, Severity::Note);
    assert!(report.findings[0].message.contains("scalar"));
}

// ============================================================================
// The probes
// ============================================================================

/// Only devices that answered are probed.
///
/// A capability answer for hardware that is not attached is a claim the
/// machine cannot honour; the fact that the backend is compiled in is already
/// reported one section up.
#[test]
fn probes_cover_the_devices_that_answered_and_no_others() {
    let host = FakeHost {
        compiled: vec![DeviceKind::Cpu, DeviceKind::Cuda, DeviceKind::Wgpu],
        present: vec![DeviceId::cpu(), DeviceId::wgpu(0)],
        ..FakeHost::healthy()
    };
    let report = Report::gather(&host);

    let devices: Vec<&str> = report
        .probes
        .iter()
        .map(|probe| probe.device.as_str())
        .collect();
    assert!(devices.contains(&"cpu"));
    assert!(devices.contains(&"wgpu"));
    assert!(
        !devices.contains(&"cuda"),
        "cuda is compiled in but absent, so nothing can be claimed about it"
    );
}

/// A machine with nothing attached probes nothing, and says so.
#[test]
fn a_machine_with_no_device_probes_nothing() {
    let host = FakeHost {
        present: Vec::new(),
        ..FakeHost::healthy()
    };
    let report = Report::gather(&host);

    assert!(report.probes.is_empty());
    assert!(report.to_text().contains("none: no device answered"));
}

/// The probe list reports rejection as well as support.
///
/// A probe section that only ever prints `native` proves the registry was
/// asked, not that it was listened to. `f64` reduction is unsupported on every
/// registry today, so one row must come back rejected with a reason.
#[test]
fn a_probe_reports_the_registrys_reason_for_a_rejection() {
    let report = Report::gather(&FakeHost::healthy());

    let rejected: Vec<_> = report
        .probes
        .iter()
        .filter(|probe| probe.support == "unsupported")
        .collect();
    assert!(
        !rejected.is_empty(),
        "no probe exercises the rejection path"
    );
    for probe in rejected {
        let reason = probe
            .reason
            .as_deref()
            .expect("a rejection carries a reason");
        assert!(
            reason.contains(&probe.operation),
            "{reason:?} does not name {}",
            probe.operation
        );
    }

    let supported: Vec<_> = report
        .probes
        .iter()
        .filter(|probe| probe.support != "unsupported")
        .collect();
    assert!(!supported.is_empty(), "no probe exercises the support path");
    for probe in supported {
        assert!(
            probe.reason.is_none(),
            "a supported probe carries no reason"
        );
    }
}

/// A rejected probe is reported, but it is not a *finding*.
///
/// `f16` matmul and `f64` reduction are unsupported on the CPU registry, which
/// is how the CPU backend is rather than something wrong with the machine.
/// Making each one a note put two of them at the top of every healthy report,
/// duplicating the probe section verbatim. This pins the decision, because the
/// obvious "improvement" is to add them back.
#[test]
fn an_unsupported_operation_is_reported_without_becoming_a_finding() {
    let report = Report::gather(&FakeHost::healthy());

    assert!(
        report
            .probes
            .iter()
            .any(|probe| probe.support == "unsupported"),
        "the baseline machine must have at least one rejected probe"
    );
    assert_eq!(codes(&report), Vec::<&str>::new());
}

// ============================================================================
// The subcommand
// ============================================================================

#[test]
fn the_argument_vocabulary_is_closed() {
    let (text, code) = run(&["--jsonn".to_string()]);
    assert_eq!(code, EXIT_USAGE);
    assert!(text.contains("unknown argument"), "{text}");
    assert!(text.contains("--json"), "the error names what is accepted");

    // A usage error is a different exit code from a bad report, because a CI
    // job gating on the doctor has to tell "this machine cannot run incin"
    // from "you typed the flag wrong".
    assert_ne!(EXIT_USAGE, EXIT_FINDINGS);
}

/// The subcommand renders text by default, JSON on request, and exits with
/// whatever the report concluded.
///
/// The exit code is compared against the report rather than pinned to
/// [`EXIT_OK`], because it is not always zero and the case where it is not is
/// a build this repository's feature powerset actually produces: with neither
/// `cpu` nor a GPU feature, `no-backend-compiled` fires and the doctor exits
/// non-zero. Pinning zero made that build fail this test, which is the
/// diagnostic working rather than the test finding a bug.
#[test]
fn the_subcommand_renders_text_by_default_and_json_on_request() {
    let expected = Report::gather(&HostMachine).exit_code();

    let (text, code) = run(&[]);
    assert_eq!(code, expected);
    assert!(text.starts_with("[toolchain]"), "{text}");

    let (json, code) = run(&["--json".to_string()]);
    assert_eq!(code, expected);
    let parsed: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    assert_eq!(parsed["schema_version"], SCHEMA_VERSION);
}

/// A build with no backend feature is the one configuration that exits
/// non-zero, and the feature powerset builds it.
///
/// Everything else about `no-backend-compiled` is asserted against a mocked
/// host. This asserts it against a real one, in the build where it is true —
/// so the error path is exercised by CI rather than only described.
#[test]
fn a_real_build_agrees_with_the_mock_about_whether_it_can_run() {
    let report = Report::gather(&HostMachine);
    let any_backend = cfg!(any(feature = "cpu", feature = "cuda", feature = "wgpu"));

    let claims_none = report
        .findings
        .iter()
        .any(|finding| finding.code == "no-backend-compiled");

    assert_eq!(claims_none, !any_backend);
    assert_eq!(
        report.exit_code(),
        if any_backend { EXIT_OK } else { EXIT_FINDINGS }
    );
}

// ============================================================================
// The real machine
// ============================================================================

/// The whole path works against actual hardware.
///
/// Deliberately weak on content — this runner's configuration is not something
/// to assert — and strong on the two things that must hold everywhere: the CPU
/// backend is compiled in and answers, and nothing in the report is empty
/// where it cannot be.
#[test]
fn the_real_machine_produces_a_report() {
    let report = Report::gather(&HostMachine);

    assert_eq!(report.schema_version, SCHEMA_VERSION);
    assert!(!report.toolchain.incin.is_empty());
    assert!(!report.features.is_empty());
    assert_eq!(report.devices.len(), 3);

    let cpu = report
        .devices
        .iter()
        .find(|device| device.kind == "cpu")
        .expect("the report always has a cpu row");
    assert_eq!(cpu.compiled_in, cfg!(feature = "cpu"));
    assert_eq!(cpu.available, cfg!(feature = "cpu"));

    let text = report.to_text();
    for section in [
        "[toolchain]",
        "[features]",
        "[cpu]",
        "[devices]",
        "[caches]",
        "[probes]",
        "[findings]",
    ] {
        assert!(text.contains(section), "{section} missing from\n{text}");
    }
}

/// The hand-written feature list must not fall behind `Cargo.toml`.
///
/// There is no way to enumerate one's own Cargo features at runtime, so
/// `HostMachine::features` is a list someone typed, and a feature added later
/// would silently never appear in any report. This reads the manifest and
/// fails when the two disagree, in either direction.
#[test]
fn the_reported_features_are_exactly_the_manifests() {
    let manifest =
        std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"))
            .expect("this crate has a manifest");

    let mut declared: Vec<String> = manifest
        .split("\n[features]\n")
        .nth(1)
        .expect("the manifest has a [features] table")
        .lines()
        .take_while(|line| !line.starts_with('['))
        .filter_map(|line| line.split_once('=').map(|(name, _)| name.trim()))
        // `default` is a feature list, not a capability anyone would look for
        // in a diagnostic report.
        .filter(|name| *name != "default")
        .map(ToString::to_string)
        .collect();
    declared.sort();

    let mut reported: Vec<String> = HostMachine
        .features()
        .into_iter()
        .map(|feature| feature.name)
        .collect();
    reported.sort();

    assert_eq!(
        reported, declared,
        "doctor::compiled_features and crates/incin/Cargo.toml disagree about the feature list"
    );
}

// ============================================================================
// Read-only
// ============================================================================

/// Sec. 2.3: "The command is read-only unless invoked with an explicit
/// cache-cleaning or benchmark flag." There is no such flag, so it is
/// read-only, and the place that would break that is the cache section — the
/// obvious way to answer "is this writable" is to write to it.
#[test]
fn inspecting_a_cache_path_does_not_create_it() {
    let path = std::env::temp_dir().join(format!(
        "incin-doctor-absent-{}-{}",
        std::process::id(),
        line!()
    ));
    assert!(!path.exists(), "the fixture path must start absent");

    assert_eq!(cache_state(&path), CacheState::Absent);
    assert!(
        !path.exists(),
        "the doctor created {} while reporting on it",
        path.display()
    );
}

#[test]
#[cfg(unix)]
fn a_directory_without_its_write_bit_reports_as_not_writable() {
    use std::os::unix::fs::PermissionsExt;

    let path = std::env::temp_dir().join(format!("incin-doctor-ro-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&path);
    std::fs::create_dir_all(&path).expect("a temp directory can be created");

    assert_eq!(cache_state(&path), CacheState::Writable);

    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o555))
        .expect("permissions can be changed");
    let observed = cache_state(&path);

    // Restore before asserting, so a failure does not leave an undeletable
    // directory behind.
    let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755));
    let _ = std::fs::remove_dir_all(&path);

    assert_eq!(observed, CacheState::NotWritable);
}
