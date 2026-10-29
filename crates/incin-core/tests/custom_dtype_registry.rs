//! A custom dtype that survives a checkpoint.
//!
//! `DTypeKey` holds `&'static str` and a key read off the wire holds owned
//! `String`s, so deserializing one means finding the `'static` descriptor it
//! names. Built-ins have a match arm for that. Everything else had nothing,
//! and the refusal said so in as many words. `DTypeRegistry` is what the
//! message was asking for.
//!
//! Every test here registers under a name of its own. The registry is
//! process-wide and the harness runs tests on several threads, so two tests
//! sharing a key would be two tests racing.

extern crate incin_core as incin;

use incin_core::error::Error;
use incin_core::tensor::dtype::{
    DTypeDescriptor, DTypeKey, DTypeKind, DTypeRegistry, StorageEncoding,
};

/// A descriptor in a third-party namespace, parameterized by name so each
/// test owns its own corner of the process-wide table.
fn custom(name: &'static str) -> DTypeDescriptor {
    DTypeDescriptor::new(
        DTypeKey::new("company.example", name, 1),
        DTypeKind::Float,
        StorageEncoding::scalar(2, 2),
    )
}

fn round_trip(key: DTypeKey) -> Result<DTypeKey, serde_json::Error> {
    let wire = serde_json::to_string(&key).expect("a key always serializes");
    serde_json::from_str::<DTypeKey>(&wire)
}

#[test]
fn an_unregistered_key_is_refused_and_the_message_says_what_to_do() {
    let descriptor = custom("unregistered_probe");

    // Serializing was never the problem: the key writes its three components
    // and always could. Reading them back is where the `'static` descriptor
    // has to come from somewhere.
    let error = round_trip(descriptor.key()).expect_err("no registration, no read");
    let rendered = error.to_string();
    assert!(
        rendered.contains("DTypeRegistry::register"),
        "the refusal should name the call that fixes it, got: {rendered}"
    );
}

#[test]
fn a_custom_dtype_round_trips_once_registered() {
    let descriptor = custom("round_trip");
    let key = descriptor.key();

    round_trip(key).expect_err("before registration");
    DTypeRegistry::register(descriptor).expect("registration");
    let recovered = round_trip(key).expect("after registration");

    assert_eq!(recovered, key);
    // The recovered key carries the registered descriptor's `'static` strings,
    // not the wire's owned ones, which is the whole reason the lookup exists.
    assert_eq!(recovered.namespace(), "company.example");
    assert_eq!(recovered.name(), "round_trip");
    assert_eq!(recovered.version(), 1);
    assert_eq!(DTypeRegistry::lookup(key), Some(descriptor));
}

#[test]
fn registering_the_same_descriptor_twice_is_idempotent() {
    // Two crates depending on a third that registers its own dtype must not
    // have to order themselves.
    let descriptor = custom("idempotent");
    DTypeRegistry::register(descriptor).expect("first");
    DTypeRegistry::register(descriptor).expect("second, identical");
    assert_eq!(DTypeRegistry::lookup(descriptor.key()), Some(descriptor));
}

#[test]
fn one_key_with_two_descriptors_is_refused_rather_than_replaced() {
    // The failure this prevents is not a load error but a silent one: the
    // second caller's tensors read with the first caller's encoding.
    let first = custom("disagreement");
    let second = DTypeDescriptor::new(first.key(), DTypeKind::Float, StorageEncoding::scalar(4, 4));
    DTypeRegistry::register(first).expect("first");

    let error = DTypeRegistry::register(second).expect_err("a different encoding for one key");
    assert!(
        matches!(error, Error::DTypeRegistration { reason, .. } if reason.contains("different")),
        "expected a registration refusal, got {error}"
    );
    // The first registration still stands. A refused write that half-applied
    // would be worse than the replacement it refused.
    assert_eq!(DTypeRegistry::lookup(first.key()), Some(first));
}

#[test]
fn the_incin_namespace_is_reserved() {
    let shadow = DTypeDescriptor::new(
        DTypeKey::new("incin", "f32", 1),
        DTypeKind::Quantized,
        StorageEncoding::scalar(1, 1),
    );
    let error = DTypeRegistry::register(shadow).expect_err("incin is not yours");
    assert!(
        matches!(error, Error::DTypeRegistration { reason, .. } if reason.contains("reserved")),
        "expected a reserved-namespace refusal, got {error}"
    );
}

#[test]
fn a_builtin_key_needs_no_registration_and_is_not_stored() {
    let f32_key = <f32 as incin_core::tensor::dtype::ConstDType>::DESCRIPTOR.key();
    assert_eq!(round_trip(f32_key).expect("built-ins always read"), f32_key);
    // Built-ins resolve through their match arms. Storing them here too would
    // give one key two sources of truth.
    assert_eq!(DTypeRegistry::lookup(f32_key), None);
}
