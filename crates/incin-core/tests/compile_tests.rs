//! The compile-fail suite, plus the check that it fails for the right reasons.
//!
//! `trybuild` compares each case against its recorded `.stderr` and passes when
//! they match. It has no opinion about *what* the error was, so a case whose
//! source has rotted — a broken import, a path to a module that has since become
//! private — keeps passing while asserting nothing about the rule it was written
//! for. `SHP-007` found four such cases in this directory, one of which
//! (`macro_module_invalid`) produced no error from the macro at all.
//!
//! [`compile_fail_cases_fail_for_their_stated_reason`] is the guard against that
//! happening again. It is shared with `mesh_compile`, which has its own
//! directory for the reason recorded in [`support`].

mod support;

use std::collections::BTreeMap;
use std::path::Path;

#[test]
/// Compile fail.
fn compile_fail() {
    let t = trybuild::TestCases::new();
    t.pass("tests/compile_pass/*.rs");
    t.compile_fail("tests/compile_fail/*.rs");
}

/// Text that must appear in a case's recorded output, keyed by file stem.
///
/// An error code where one exists, because that is the specific thing the case
/// is pinning: `E0624` is "this associated function is private" and nothing
/// else. Macro rejections have no code, so those are matched on the message the
/// macro emits.
fn expected_reasons() -> BTreeMap<&'static str, &'static str> {
    BTreeMap::from([
        // Shape rules, all unsatisfied trait bounds.
        ("concat_static_mismatch", "E0277"),
        ("checked_byte_len_is_private", "E0423"),
        ("checked_numel_is_private", "E0423"),
        ("construction_witness_is_private", "E0624"),
        ("conv2d_invalid_shape", "E0277"),
        ("forward_batchnorm_mismatch", "E0277"),
        ("forward_broadcast_mismatch", "E0277"),
        ("forward_conv1d_static_mismatch", "E0277"),
        ("forward_conv2d_static_mismatch", "E0277"),
        ("forward_linear_partial_mismatch", "E0277"),
        ("forward_linear_static_mismatch", "E0277"),
        ("forward_model_building_mismatch", "E0277"),
        ("execution_request_requires_validated", "E0308"),
        ("execute_is_not_blanket", "E0277"),
        ("flatten_invalid_range", "Invalid flatten range"),
        ("kernel_conv2d_channel_mismatch", "E0277"),
        ("layer_builder_invalid_count", "E0277"),
        ("layer_builder_invalid_order", "E0277"),
        ("matmul_static_mismatch", "E0277"),
        ("matmul_rank8_static_mismatch", "E0277"),
        ("named_dim_concat_mismatch", "E0277"),
        ("reshape_static_mismatch", "E0277"),
        // Shape rules the compiler reports as a type mismatch rather than an
        // unmet bound, because the shape appears in the signature directly.
        ("device_mismatch", "E0308"),
        ("dtype_mismatch", "E0308"),
        ("dyn_is_a_unit_marker", "E0618"),
        ("gradients_backend_payload_is_private", "E0423"),
        ("loss_cross_entropy_mismatch", "E0308"),
        ("named_dim_identity_mismatch", "E0308"),
        ("named_dim_size_mismatch", "E0308"),
        ("shape_mismatch", "E0308"),
        ("stack_static_mismatch", "E0308"),
        ("tensor_meta_fields_are_private", "E0451"),
        // Proof lowering: the seal around `Validated` and the descriptor
        // taxonomy (`EXE-002`), and the frontend binding (`EXE-003`).
        ("operation_spec_is_sealed", "E0277"),
        ("shape_rule_needs_the_frontend_proof", "E0277"),
        ("validated_fields_are_private", "E0451"),
        ("validated_new_is_crate_private", "E0624"),
        ("unsupported_dtype_backend_pair", "E0277"),
        // `EXE-009`: an operation family with no unsupported default, so a
        // backend that omits a method fails to compile instead of answering
        // the call with an error at run time.
        ("module_ops_has_no_unsupported_default", "E0046"),
        ("tensor_ops_has_no_unsupported_default", "E0046"),
        // Macros, which emit their own diagnostics and carry no error code.
        ("macro_idx_invalid", "expected `..=`"),
        (
            "macro_module_invalid",
            "unknown attribute argument for #[module]",
        ),
        ("macro_s_invalid", "expected identifier"),
    ])
}

#[test]
fn compile_fail_cases_fail_for_their_stated_reason() {
    support::compile_fail_cases_name_their_reason(
        Path::new("tests/compile_fail"),
        &expected_reasons(),
    );
}
