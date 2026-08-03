//! `UX-008`: Reproducibility manifest replay and incompatibility diffs test.

#![cfg(feature = "compiled")]

use incin_core::experimental::compiled::ReproducibilityManifest;

#[test]
fn test_manifest_serialization_roundtrip() {
    let manifest = ReproducibilityManifest::new(
        42,
        "f32",
        "mesh[dp=2,tp=2]",
        "linux-x86_64",
        "plan_hash_abcdef123456",
    );

    let json = manifest
        .to_json()
        .expect("Serialization to JSON must succeed");
    let restored =
        ReproducibilityManifest::from_json(&json).expect("Deserialization from JSON must succeed");

    assert_eq!(manifest, restored);
    assert!(manifest.replay_diff(&restored).is_empty());
}

#[test]
fn test_manifest_replay_diff_detects_incompatibilities() {
    let manifest1 = ReproducibilityManifest::new(
        42,
        "f32",
        "mesh[dp=2,tp=2]",
        "linux-x86_64",
        "plan_hash_abcdef123456",
    );

    let manifest2 = ReproducibilityManifest::new(
        999,               // Mismatched seed
        "bf16",            // Mismatched precision
        "mesh[dp=4,tp=1]", // Mismatched mesh
        "linux-x86_64",
        "plan_hash_different", // Mismatched plan hash
    );

    let diffs = manifest1.replay_diff(&manifest2);
    assert_eq!(diffs.len(), 4);
    assert!(diffs.iter().any(|d| d.contains("Mismatched random seed")));
    assert!(
        diffs
            .iter()
            .any(|d| d.contains("Mismatched precision policy"))
    );
    assert!(diffs.iter().any(|d| d.contains("Mismatched mesh topology")));
    assert!(diffs.iter().any(|d| d.contains("Mismatched plan hash")));
}
