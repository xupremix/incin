#![cfg(feature = "autotune")]

use incin_backends::tuning::{
    cache::CacheKey,
    identity::{
        CompilerFingerprint, DeviceFingerprint, SoftwareVersion, TuningEnvironmentFingerprint,
    },
    service::{
        AutotunePolicy, DisabledTuning, SelectionSource, TuningCandidate, TuningContext,
        TuningScope, TuningService,
    },
    telemetry::{TuningExplain, TuningProvenance, emit_tuning_telemetry},
};
use incin_core::{exec::Determinism, prelude::Cuda};
use std::time::Duration;

fn static_environment() -> TuningEnvironmentFingerprint<Cuda> {
    TuningEnvironmentFingerprint::new(
        DeviceFingerprint::<Cuda>::new(
            "GPU-11111111-2222-3333-4444-555555555555",
            "sm_90",
            SoftwareVersion::new(12, 8, 0),
        )
        .unwrap(),
        CompilerFingerprint::<Cuda>::new(
            "nvrtc",
            SoftwareVersion::new(12, 8, 0),
            "sm_90",
            &["default-options"],
        )
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn tuning_provenance_and_explain_formatting() {
    let env = static_environment();
    let key = CacheKey::new_dyn("kernel", &env.clone().erase(), "matmul_f32_tile").unwrap();
    let candidate = TuningCandidate::new(12345, "block=256,vector_width=4", true, 0).unwrap();
    let disabled = TuningService::<DisabledTuning>::disabled();
    let context = TuningContext::<Cuda, _>::kernel(
        env.clone(),
        Determinism::Permitted,
        1024,
        Duration::from_secs(1),
    )
    .unwrap();

    let decision = disabled
        .decide(&context, key.clone(), &[candidate.clone()], 12345, 12345)
        .unwrap();

    let selection = match decision {
        incin_backends::tuning::service::ServiceDecision::Selected(sel) => sel,
        _ => panic!("expected selected candidate"),
    };

    let provenance = TuningProvenance::new(
        key.clone(),
        TuningScope::Kernel,
        env.erase(),
        0xabc123,
        &selection,
    );

    assert_eq!(provenance.winner_hash, 12345);
    assert_eq!(provenance.source, SelectionSource::DisabledFallback);
    assert_eq!(provenance.winner_encoding, "block=256,vector_width=4");

    let explain = TuningExplain::new(provenance.clone(), AutotunePolicy::Disabled, 4, 2);

    let text = explain.explain_text();
    assert!(text.contains("Tuning Explain [Kernel]"));
    assert!(text.contains("matmul_f32_tile"));
    assert!(text.contains("DisabledFallback"));
    assert!(text.contains("GPU-11111111-2222-3333-4444-555555555555"));

    let json = explain.explain_json();
    assert!(json.contains("\"scope\":\"Kernel\""));
    assert!(json.contains("\"problem\":\"matmul_f32_tile\""));
    assert!(json.contains("\"source\":\"DisabledFallback\""));
    assert!(json.contains("\"winner_hash\":\"0x3039\""));

    emit_tuning_telemetry(1, &provenance);
}
