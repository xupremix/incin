use super::*;

/// Counts derived directly from the canonical operation catalog.
pub struct OperationCoverage {
    pub canonical: usize,
    pub backend_executable: usize,
    pub non_backend_executable: usize,
    pub by_site: [(ExecutionSite, usize); 7],
}

/// Return operation coverage without maintaining a second count.
#[must_use]
pub fn operation_coverage() -> OperationCoverage {
    let mut by_site = [
        (ExecutionSite::Kernel, 0),
        (ExecutionSite::Creation, 0),
        (ExecutionSite::HostReadback, 0),
        (ExecutionSite::Composed, 0),
        (ExecutionSite::Mutation, 0),
        (ExecutionSite::DeviceTransfer, 0),
        (ExecutionSite::GraphState, 0),
    ];
    let mut backend_executable = 0;
    for row in OPERATION_CATALOG {
        if row.site.is_backend_executable() {
            backend_executable += 1;
        }
        if let Some((_, count)) = by_site.iter_mut().find(|(site, _)| *site == row.site) {
            *count += 1;
        }
    }
    OperationCoverage {
        canonical: OPERATION_CATALOG.len(),
        backend_executable,
        non_backend_executable: OPERATION_CATALOG.len() - backend_executable,
        by_site,
    }
}

/// Render the operation coverage report from the canonical catalog.
#[must_use]
pub fn operation_coverage_document() -> alloc::string::String {
    use core::fmt::Write;

    let coverage = operation_coverage();
    let mut document = alloc::string::String::from(
        "# Canonical operation coverage\n\nThis file is generated from `incin_core::exec::OPERATION_CATALOG`; the Rust catalog is authoritative.\n\n",
    );
    let _ = writeln!(document, "- Canonical operations: {}", coverage.canonical);
    let _ = writeln!(
        document,
        "- Backend-executable operations: {}",
        coverage.backend_executable
    );
    let _ = writeln!(
        document,
        "- Non-backend execution sites: {}\n",
        coverage.non_backend_executable
    );
    document.push_str("| Execution site | Operations |\n|---|---:|\n");
    for (site, count) in coverage.by_site {
        let _ = writeln!(document, "| `{site:?}` | {count} |");
    }
    document
        .push_str("\n## Non-backend operations\n\n| Operation | Site | Reason |\n|---|---|---|\n");
    for row in OPERATION_CATALOG {
        if let Some(reason) = row.site.blocking_reason() {
            let _ = writeln!(
                document,
                "| `{}` | `{:?}` | {} |",
                row.name, row.site, reason
            );
        }
    }
    document
}
