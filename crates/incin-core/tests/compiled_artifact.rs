#![cfg(feature = "compiled")]

use incin_core::experimental::compiled::{
    ArtifactVersion, CapturedGraph, CompileOptions, CompiledArtifact, CompiledPlan,
};
use incin_core::graph::Graph;
use incin_core::prelude::DTypeId;
use incin_core::prelude::OperationKind;
use std::collections::BTreeMap;

fn make_test_plan() -> CompiledPlan {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![4], DTypeId::F32, Some("y".into()));
    graph.mark_input(x);
    graph.mark_output(y);
    graph.add_node(OperationKind::Relu, vec![x], vec![y], BTreeMap::new());
    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    CompiledPlan::compile(captured, CompileOptions::new())
}

fn current_version() -> ArtifactVersion {
    ArtifactVersion::new(0, 1, 0)
}

#[test]
fn test_artifact_roundtrip() {
    let plan = make_test_plan();
    let artifact = CompiledArtifact::new(plan, current_version(), "test_artifact".into())
        .expect("artifact creation should succeed");
    let bytes = artifact.serialize().expect("serialization should succeed");
    let loaded = CompiledArtifact::deserialize(&bytes).expect("deserialization should succeed");
    assert_eq!(artifact, loaded);
}

#[test]
fn test_artifact_integrity_valid() {
    let plan = make_test_plan();
    let artifact = CompiledArtifact::new(plan, current_version(), "integrity_test".into())
        .expect("artifact creation should succeed");
    artifact
        .verify_integrity()
        .expect("integrity check should pass for unmodified artifact");
}

#[test]
fn test_artifact_corruption_detected() {
    let plan = make_test_plan();
    let artifact = CompiledArtifact::new(plan, current_version(), "corruption_test".into())
        .expect("artifact creation should succeed");
    let mut bytes = artifact.serialize().expect("serialization should succeed");

    // Corrupt the last few bytes of the payload
    let len = bytes.len();
    if len > 10 {
        bytes[len - 5] ^= 0xFF;
    }

    // Deserialize will succeed (JSON may still parse), but integrity must fail
    let result = CompiledArtifact::deserialize(&bytes);
    match result {
        Ok(corrupted) => {
            // If it deserialized, integrity check must catch the corruption
            assert!(
                corrupted.verify_integrity().is_err(),
                "corrupted artifact should fail integrity check"
            );
        }
        Err(_) => {
            // Deserialization failure is also acceptable for corrupted bytes
        }
    }
}

#[test]
fn test_artifact_compatibility_same_version() {
    let plan = make_test_plan();
    let version = current_version();
    let artifact = CompiledArtifact::new(plan, version.clone(), "compat_test".into())
        .expect("artifact creation should succeed");
    artifact
        .check_compatibility(&version)
        .expect("same version should be compatible");
}

#[test]
fn test_artifact_incompatible_major_version() {
    let plan = make_test_plan();
    let version = ArtifactVersion::new(0, 1, 0);
    let artifact = CompiledArtifact::new(plan, version, "incompat_test".into())
        .expect("artifact creation should succeed");
    let required = ArtifactVersion::new(1, 0, 0); // different major
    assert!(
        artifact.check_compatibility(&required).is_err(),
        "different major version should be incompatible"
    );
}

#[test]
fn test_artifact_load_happy_path() {
    let plan = make_test_plan();
    let version = current_version();
    let artifact = CompiledArtifact::new(plan, version.clone(), "load_test".into())
        .expect("artifact creation should succeed");
    let bytes = artifact.serialize().expect("serialization should succeed");
    let loaded = CompiledArtifact::load(&bytes, &version).expect("load should succeed");
    assert_eq!(loaded.header.label, "load_test");
}
