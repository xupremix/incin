#![cfg(feature = "autotune")]

use incin_backends::tuning::{
    cache::{
        CacheError, CacheKey, CacheLimits, CacheRecord, MeasurementMethod, PersistentTuningCache,
    },
    identity::{
        CompilerFingerprint, DeviceFingerprint, IdentityError, SoftwareVersion,
        TuningEnvironmentFingerprint,
    },
};
use incin_core::prelude::{Cpu, Cuda, Dyn};
use serde_json::{Value, json};
use std::{fs, sync::Arc, thread, time::Duration};

const DRIVER: SoftwareVersion = SoftwareVersion::new(12, 8, 0);

fn static_environment() -> TuningEnvironmentFingerprint<Cuda> {
    TuningEnvironmentFingerprint::new(
        DeviceFingerprint::<Cuda>::new("GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "sm_90", DRIVER)
            .unwrap(),
        CompilerFingerprint::<Cuda>::new(
            "nvrtc",
            SoftwareVersion::new(12, 8, 0),
            "sm_90",
            &["default-math"],
        )
        .unwrap(),
    )
    .unwrap()
}

fn dynamic_environment() -> TuningEnvironmentFingerprint<Dyn> {
    static_environment().erase()
}

fn key(problem: &str) -> CacheKey<Dyn> {
    CacheKey::new_dyn("kernel", &dynamic_environment(), problem).unwrap()
}

fn record(problem: &str, created: u64, winner: &str) -> CacheRecord {
    CacheRecord::new_at(
        key(problem),
        created,
        MeasurementMethod::Measured,
        7,
        Some(123),
        0xa11c_e5e7,
        winner,
    )
    .unwrap()
}

fn generous_limits() -> CacheLimits {
    CacheLimits::new(32, 128 * 1024, None).unwrap()
}

#[test]
fn static_and_dyn_keys_have_parity_and_runtime_projection_checks_backend() {
    let static_key = CacheKey::<Cuda>::new("kernel", &static_environment(), "neg:f32").unwrap();
    let dynamic_key =
        CacheKey::<Dyn>::new_dyn("kernel", &dynamic_environment(), "neg:f32").unwrap();
    assert_eq!(static_key.digest(), dynamic_key.digest());
    assert_eq!(
        dynamic_key
            .clone()
            .try_into_static::<Cuda>()
            .unwrap()
            .digest(),
        static_key.digest()
    );
    assert!(matches!(
        dynamic_key.try_into_static::<Cpu>(),
        Err(IdentityError::BackendMismatch {
            expected: "cpu",
            actual: "cuda"
        })
    ));
}

#[test]
fn atomic_round_trip_preserves_provenance_and_legal_set_guard() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("tuning.json");
    let mut cache = PersistentTuningCache::open_at(&path, generous_limits(), 10).unwrap();
    let expected = record("neg:f32:bucket=12", 10, "block=256,access=packed4");
    cache.commit_at(expected.clone(), 10).unwrap();

    let bytes = fs::read(&path).unwrap();
    assert!(bytes.starts_with(b"{"));
    let reopened = PersistentTuningCache::open_at(&path, generous_limits(), 11).unwrap();
    let loaded = reopened
        .lookup(expected.key(), expected.legal_candidates_digest())
        .unwrap();
    assert_eq!(loaded, &expected);
    assert!(
        reopened
            .lookup(expected.key(), expected.legal_candidates_digest() ^ 1)
            .is_none(),
        "a cache import is not valid for a changed legal candidate set"
    );
    assert!(fs::read_dir(directory.path()).unwrap().all(|entry| {
        !entry
            .unwrap()
            .file_name()
            .to_string_lossy()
            .contains(".tmp-")
    }));
}

#[test]
fn unknown_fields_are_tolerated_without_weakening_checksum_validation() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("tuning.json");
    let mut cache = PersistentTuningCache::open_at(&path, generous_limits(), 10).unwrap();
    let expected = record("sum:f32", 10, "block=128");
    cache.commit_at(expected.clone(), 10).unwrap();

    let mut document: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    document["future_top_level_field"] = json!({"understood_by": "schema-v2"});
    document["entries"][0]["future_record_field"] = json!(17);
    fs::write(&path, serde_json::to_vec_pretty(&document).unwrap()).unwrap();

    let reopened = PersistentTuningCache::open_at(&path, generous_limits(), 11).unwrap();
    assert!(reopened.recovery().is_none());
    assert!(
        reopened
            .lookup(expected.key(), expected.legal_candidates_digest())
            .is_some()
    );

    document["entries"][0]["winner"] = json!("tampered");
    fs::write(&path, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
    let rejected = PersistentTuningCache::open_at(&path, generous_limits(), 12).unwrap();
    assert!(rejected.is_empty());
    assert!(rejected.recovery().unwrap().reason().contains("checksum"));
}

#[test]
fn malformed_schema_and_invalid_records_are_quarantined() {
    for (name, bytes, reason) in [
        ("malformed", b"{not-json".to_vec(), "invalid JSON"),
        (
            "schema",
            serde_json::to_vec_pretty(&json!({
                "schema": 999,
                "entries": [],
                "checksum": "0000000000000000"
            }))
            .unwrap(),
            "schema 999",
        ),
    ] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join(format!("{name}.json"));
        fs::write(&path, bytes).unwrap();
        let cache = PersistentTuningCache::open_at(&path, generous_limits(), 10).unwrap();
        let recovery = cache.recovery().unwrap();
        assert!(recovery.reason().contains(reason));
        assert!(recovery.quarantined_path().exists());
        assert!(!path.exists());
    }

    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("invalid-record.json");
    let mut cache = PersistentTuningCache::open_at(&path, generous_limits(), 10).unwrap();
    cache
        .commit_at(record("valid", 10, "block=128"), 10)
        .unwrap();
    let mut document: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    document["entries"][0]["winner"] = json!("");
    // The checksum is intentionally stale. Either checksum or record
    // validation must quarantine the untrusted import before use.
    fs::write(&path, serde_json::to_vec_pretty(&document).unwrap()).unwrap();
    let cache = PersistentTuningCache::open_at(&path, generous_limits(), 11).unwrap();
    assert!(cache.is_empty());
    assert!(cache.recovery().is_some());
}

#[test]
fn count_and_age_bounds_evict_deterministically() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("bounded.json");
    let limits = CacheLimits::new(2, 128 * 1024, Some(Duration::from_millis(100))).unwrap();
    let mut cache = PersistentTuningCache::open_at(&path, limits, 100).unwrap();
    cache.commit_at(record("oldest", 100, "a"), 100).unwrap();
    cache.commit_at(record("middle", 150, "b"), 150).unwrap();
    cache.commit_at(record("newest", 175, "c"), 175).unwrap();
    assert_eq!(cache.len(), 2);
    assert!(cache.lookup(&key("oldest"), 0xa11c_e5e7).is_none());
    assert!(cache.lookup(&key("middle"), 0xa11c_e5e7).is_some());
    assert!(cache.lookup(&key("newest"), 0xa11c_e5e7).is_some());

    let expired = PersistentTuningCache::open_at(&path, limits, 276).unwrap();
    assert!(expired.is_empty());
    let reopened = PersistentTuningCache::open_at(&path, limits, 276).unwrap();
    assert!(
        reopened.is_empty(),
        "age pruning must be persisted atomically"
    );
}

#[test]
fn byte_bound_evicts_old_records_and_rejects_an_oversized_winner() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("bytes.json");
    let limits = CacheLimits::new(32, 900, None).unwrap();
    let mut cache = PersistentTuningCache::open_at(&path, limits, 10).unwrap();
    cache
        .commit_at(record("old", 10, &"a".repeat(250)), 10)
        .unwrap();
    cache
        .commit_at(record("new", 20, &"b".repeat(250)), 20)
        .unwrap();
    assert!(fs::metadata(&path).unwrap().len() <= 900);
    assert!(cache.lookup(&key("new"), 0xa11c_e5e7).is_some());

    let oversized = record("too-large", 30, &"z".repeat(4000));
    assert!(matches!(
        cache.commit_at(oversized, 30),
        Err(CacheError::RecordTooLarge { maximum_bytes: 900 })
    ));
    let reopened = PersistentTuningCache::open_at(&path, limits, 31).unwrap();
    assert!(reopened.lookup(&key("new"), 0xa11c_e5e7).is_some());
}

#[test]
fn concurrent_process_views_merge_under_the_file_lock() {
    let directory = Arc::new(tempfile::tempdir().unwrap());
    let path = directory.path().join("shared.json");
    let mut first = PersistentTuningCache::open_at(&path, generous_limits(), 10).unwrap();
    let mut second = PersistentTuningCache::open_at(&path, generous_limits(), 10).unwrap();

    let first_thread = thread::spawn(move || {
        first
            .commit_at(record("first", 10, "block=128"), 10)
            .unwrap();
    });
    let second_thread = thread::spawn(move || {
        second
            .commit_at(record("second", 11, "block=256"), 11)
            .unwrap();
    });
    first_thread.join().unwrap();
    second_thread.join().unwrap();

    let merged = PersistentTuningCache::open_at(&path, generous_limits(), 12).unwrap();
    assert_eq!(merged.len(), 2);
    assert!(merged.lookup(&key("first"), 0xa11c_e5e7).is_some());
    assert!(merged.lookup(&key("second"), 0xa11c_e5e7).is_some());
}

#[test]
fn measured_records_require_real_measurement_provenance() {
    assert!(matches!(
        CacheRecord::new_at(
            key("invalid"),
            10,
            MeasurementMethod::Measured,
            0,
            None,
            1,
            "block=128",
        ),
        Err(CacheError::InvalidRecord { .. })
    ));
    assert!(
        CacheRecord::new_at(
            key("heuristic"),
            10,
            MeasurementMethod::Heuristic,
            0,
            None,
            1,
            "block=128",
        )
        .is_ok()
    );
    assert!(CacheLimits::new(0, 1024, None).is_err());
    assert!(CacheLimits::new(1, 1, None).is_err());
}

#[test]
fn static_cache_contracts_are_compile_checked() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/tuning_cache_compile_fail/*.rs");
}
