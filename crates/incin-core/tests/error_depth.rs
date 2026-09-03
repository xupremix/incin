//! Errors a caller can act on without going to read the source.
//!
//! Each of these used to render as a single clause naming what went wrong and
//! nothing about what to do. The summary line is unchanged, so a message stays
//! greppable and still reads as one sentence; what follows it is the operand
//! that broke the rule, the rule itself, and the edit that satisfies it.

use incin_core::error::Error;
use incin_core::exec::catalog::DescriptorError;
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::device::DeviceId;
use incin_core::tensor::dtype::DTypeId;

/// Every widened message keeps its original first line.
fn summary(rendered: &str) -> &str {
    rendered.lines().next().unwrap_or_default()
}

#[test]
fn an_arity_error_says_how_many_operands_to_add_or_drop() {
    let rendered = DescriptorError::Arity {
        operation: OperationKind::Conv2dExact,
        expected: 2..=3,
        actual: 1,
    }
    .to_string();

    assert_eq!(summary(&rendered), "conv2d: input arity 1 is outside 2..=3");
    assert!(rendered.contains("fix: pass 1 more operand"), "{rendered}");
}

#[test]
fn a_rank_error_names_the_operand_and_the_axis_edit() {
    let rendered = DescriptorError::Rank {
        operation: OperationKind::MaxPool2d,
        input: 0,
        expected: 3..=4,
        actual: 2,
    }
    .to_string();

    assert!(rendered.contains("operand 0: rank 2"), "{rendered}");
    assert!(rendered.contains("accepts: rank 3 to 4"), "{rendered}");
    assert!(rendered.contains("unsqueeze"), "{rendered}");
}

/// The device case: which operand is where, and that nothing moves data
/// across devices on its own.
#[test]
fn a_device_mismatch_names_both_devices_and_the_move_that_fixes_it() {
    let rendered = DescriptorError::DeviceMismatch {
        operation: OperationKind::Add,
        input: 1,
        expected: DeviceId::cpu(),
        actual: DeviceId::cpu(),
    }
    .to_string();

    assert!(rendered.contains("operand 0 is on:"), "{rendered}");
    assert!(rendered.contains("operand 1 is on:"), "{rendered}");
    assert!(rendered.contains("to_device"), "{rendered}");
}

/// The exact failure a float class index produces, which is what a first
/// `cross_entropy_loss` call hits.
#[test]
fn an_index_dtype_error_says_to_build_an_integer_tensor() {
    let rendered = DescriptorError::InvalidAttribute {
        operation: OperationKind::CrossEntropyLoss,
        attribute: "index dtype",
        reason: "index metadata requires an integer dtype",
    }
    .to_string();

    assert_eq!(
        summary(&rendered),
        "cross_entropy_loss: invalid index dtype: index metadata requires an integer dtype"
    );
    assert!(rendered.contains("Tensor::<_, _, i64>"), "{rendered}");
}

/// An attribute with no tailored advice still gets a rule and a fix line
/// rather than falling back to an empty one.
#[test]
fn an_unrecognized_attribute_still_carries_a_rule_and_a_fix() {
    let rendered = DescriptorError::InvalidAttribute {
        operation: OperationKind::TopK,
        attribute: "k",
        reason: "k must be at most the axis extent",
    }
    .to_string();

    assert!(rendered.contains("rule: k must be at most the axis extent"));
    assert!(rendered.contains("fix: "), "{rendered}");
}

#[test]
fn an_unavailable_backend_names_the_cargo_feature_that_enables_it() {
    let rendered = Error::BackendUnavailable { backend: "cuda" }.to_string();

    assert_eq!(
        summary(&rendered),
        "Backend 'cuda' is unavailable in this build"
    );
    assert!(rendered.contains("cargo feature"), "{rendered}");
    assert!(rendered.contains("rebuild"), "{rendered}");
}

#[test]
fn an_unsupported_dtype_points_at_the_capability_table() {
    let rendered = Error::UnsupportedDType {
        dtype: DTypeId::Bool.descriptor(),
        backend: "Cpu",
        op: "matmul",
    }
    .to_string();

    assert!(rendered.contains("backend: Cpu"), "{rendered}");
    assert!(rendered.contains("to_dtype"), "{rendered}");
    assert!(rendered.contains("docs/capabilities.md"), "{rendered}");
}
