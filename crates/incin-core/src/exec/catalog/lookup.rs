use super::*;

/// Looks up the full catalog entry for one operation.
pub fn catalog_entry(operation: OperationKind) -> Option<&'static OperationCatalogEntry> {
    OPERATION_CATALOG
        .iter()
        .find(|row| row.operation == operation)
}

/// Explicit ONNX projection for canonical built-in operations.
/// Unsupported operations return `None` instead of being silently renamed.
pub const fn onnx_name(operation: OperationKind) -> Option<&'static str> {
    Some(match operation {
        OperationKind::Add => "Add",
        OperationKind::Sub => "Sub",
        OperationKind::Mul => "Mul",
        OperationKind::Div => "Div",
        OperationKind::MatMulExact | OperationKind::BatchedMatMul => "MatMul",
        OperationKind::Relu => "Relu",
        OperationKind::Exp => "Exp",
        OperationKind::Neg => "Neg",
        OperationKind::Sqrt => "Sqrt",
        OperationKind::Log => "Log",
        OperationKind::Tanh => "Tanh",
        OperationKind::Sigmoid => "Sigmoid",
        OperationKind::Softmax => "Softmax",
        OperationKind::ReshapeExact => "Reshape",
        OperationKind::TransposeExact => "Transpose",
        OperationKind::ConcatExact => "Concat",
        // Stack inserts a new dimension and cannot be represented by ONNX
        // Concat without an explicit unsqueeze lowering and descriptor
        // payload. Keep the projection fail closed until that lowering exists.
        OperationKind::Conv1dExact | OperationKind::Conv2dExact => "Conv",
        OperationKind::ConvTranspose2d => "ConvTranspose",
        OperationKind::MaxPool2d => "MaxPool",
        OperationKind::AvgPool2d => "AveragePool",
        OperationKind::AdaptiveAvgPool2dExact => "GlobalAveragePool",
        OperationKind::CmpEq => "Equal",
        OperationKind::CmpLt => "Less",
        OperationKind::CmpLe => "LessOrEqual",
        OperationKind::CmpGt => "Greater",
        OperationKind::CmpGe => "GreaterOrEqual",
        OperationKind::LogicalAnd => "And",
        OperationKind::LogicalOr => "Or",
        OperationKind::LogicalNot => "Not",
        OperationKind::Maximum => "Max",
        OperationKind::Minimum => "Min",
        OperationKind::WhereCond => "Where",
        OperationKind::Gather => "GatherElements",
        OperationKind::IndexSelect => "Gather",
        OperationKind::Scatter => "ScatterElements",
        OperationKind::Unsqueeze => "Unsqueeze",
        OperationKind::Repeat => "Tile",
        OperationKind::Pad => "Pad",
        OperationKind::Triu | OperationKind::Tril => "Trilu",
        OperationKind::PixelShuffle => "DepthToSpace",
        OperationKind::GroupNorm => "GroupNormalization",
        OperationKind::InstanceNorm => "InstanceNormalization",
        _ => return None,
    })
}

/// Render the human-reviewed semantics inventory from the code catalog.
#[must_use]
pub fn operation_semantics_document() -> alloc::string::String {
    use core::fmt::Write as _;
    let coverage = operation_coverage();
    let mut out = alloc::string::String::from(
        "# Canonical operation semantics\n\nThis file is generated from `incin_core::exec::OPERATION_CATALOG`; the Rust catalog is authoritative. Families classify operations and never imply backend support. `TypedContract` and `TypedInference` refer to the exact descriptor's typed attribute validator and checked inference branch; they do not permit a backend-specific default. `Site` records where the result is produced and therefore whether `Execute<O>` can carry it: `Kernel`, `Creation` and `HostReadback` can, while `Mutation`, `DeviceTransfer` and `GraphState` cannot be expressed by that trait as it currently stands.\n\n| ID | Descriptor | Attributes | Site | Input/output arity | Rank | Broadcast | Dtype/output | Empty/non-finite | Gradient | Deterministic | Layout | Legacy mapping |\n|---|---|---|---|---|---|---|---|---|---|:--:|---|---|\n",
    );
    let _ = writeln!(
        out,
        "Canonical operations: {}\nBackend-executable operations: {}\nNon-backend execution sites: {}\n",
        coverage.canonical, coverage.backend_executable, coverage.non_backend_executable
    );
    out.push_str("| Execution site | Count |\n|---|---:|\n");
    for (site, count) in coverage.by_site {
        let _ = writeln!(out, "| `{:?}` | {} |", site, count);
    }
    out.push('\n');
    for row in OPERATION_CATALOG {
        let max_arity = if *row.input_arity.end() == usize::MAX {
            alloc::string::String::from("many")
        } else {
            row.input_arity.end().to_string()
        };
        let max_output_arity = if *row.output_arity.end() == usize::MAX {
            alloc::string::String::from("many")
        } else {
            row.output_arity.end().to_string()
        };
        let _ = writeln!(
            out,
            "| `{}` | `{}` | `{}` | `{:?}` | {}-{} / {}-{} | {}-{} | `{:?}` | `{:?}` / `{:?}` | `{:?}` / `{:?}` | `{:?}` | {} | `{:?}` | `{}` |",
            row.name,
            row.descriptor,
            row.attributes,
            row.site,
            row.input_arity.start(),
            max_arity,
            row.output_arity.start(),
            max_output_arity,
            row.accepted_ranks.start(),
            row.accepted_ranks.end(),
            row.broadcasting,
            row.dtype,
            row.output,
            row.empty,
            row.numeric,
            row.gradient,
            if row.deterministic { "yes" } else { "no" },
            row.layout,
            row.legacy_source,
        );
    }
    out
}
