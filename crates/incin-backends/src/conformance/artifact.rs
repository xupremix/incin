//! The machine-readable conformance artifact (issue #83).
//!
//! The documentation generator consumes run results rather than prose claims,
//! so the matrix writes itself out as JSON. This module defines the schema,
//! the version, and the writer; the runner lives in
//! [`super::matrix`].
//!
//! # Schema (version 1)
//!
//! ```json
//! {
//!   "schema_version": 1,
//!   "summary": {
//!     "Cpu":   { "advertised": 3500, "passed": 2555, "failed": 0, "skipped": 945 },
//!     "Cuda":  { "advertised": 70, "passed": 0, "failed": 0, "skipped": 70 },
//!     "Wgpu":  { "advertised": 46, "passed": 0, "failed": 0, "skipped": 46 },
//!     "Metal": { "advertised": 25, "passed": 0, "failed": 0, "skipped": 25 }
//!   },
//!   "cells": [
//!     {
//!       "backend": "Cpu", "operation": "add", "dtype": "f32",
//!       "layout": "Contiguous", "rank": 2, "training": false,
//!       "math_mode": "Precise", "negative_probe": false,
//!       "status": "passed"
//!     },
//!     {
//!       "backend": "Cpu", "operation": "matmul", "dtype": "f32",
//!       "layout": "Strided", "rank": 2, "training": true,
//!       "math_mode": "Precise", "negative_probe": false,
//!       "status": "failed", "reason": "advertised row refused at run time: ..."
//!     }
//!   ],
//!   "skipped": [
//!     {
//!       "backend": "Cuda", "operation": "add", "skip": "no-device",
//!       "count": 12, "reason": "no-device: no executable device ...",
//!       "example": { "dtype": "f32", "layout": "Contiguous", "rank": 2,
//!                     "training": true, "math_mode": "Precise" }
//!     }
//!   ]
//! }
//! ```
//!
//! Field rules, so a consumer can rely on them:
//!
//! * `schema_version` is [`SCHEMA_VERSION`]. A consumer that does not
//!   recognize the version must refuse the file rather than guess.
//! * `cells` holds one object per *executed* tuple: every pass and every
//!   failure. `status` is `"passed"` or `"failed"`, nothing else.
//! * `skipped` holds one object per `(backend, operation, skip)` group: the
//!   count of tuples it stands for, the full reason once, and one example
//!   tuple. Grouping is what keeps the file small: skip reasons repeat
//!   identically across hundreds of tuples, and repeating them per cell would
//!   be megabytes of duplicated prose. `skip` is one of `"no-device"`,
//!   `"no-fixture"`, `"unbuildable"`.
//! * `reason` is present on every `"failed"` cell (why the backend broke its
//!   advertisement) and on every skipped group (why the harness did not
//!   execute those tuples). It is absent on `"passed"` cells, which need no
//!   explanation.
//! * `negative_probe` marks the unadvertised-dtype refusal checks. A passed
//!   probe means the executor refused what the row never advertised.
//! * Counts reconcile: for every leg, `advertised == passed + failed +
//!   skipped`, where `skipped` is the sum of its groups' counts. A consumer
//!   should check this before trusting the file.
//! * The file is kept small by construction: compact JSON, one object per
//!   executed cell, grouped skips, no tensors, no values, no per-element
//!   data.
//!
//! # Compatibility
//!
//! Bump [`SCHEMA_VERSION`] whenever a field is added, removed, or renamed,
//! and keep the old writer readable by naming the version in the file. Adding
//! an optional field is still a bump: a consumer pinned to the old version
//! would silently drop information it was never told about.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use incin_core::tensor::device::DeviceKind;

use crate::conformance::matrix::{CellVerdict, MatrixCell, MatrixReport, SkipReason};

/// The artifact schema version. See the module docs for what a bump means.
pub const SCHEMA_VERSION: u32 = 1;

/// Default artifact path, relative to the workspace root.
pub const DEFAULT_PATH: &str = "target/conformance/matrix.json";

/// Per-leg counts, as the `summary` object records them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub struct LegSummary {
    /// Cells enumerated from this leg's registry.
    pub advertised: usize,
    /// Cells that passed.
    pub passed: usize,
    /// Cells that failed.
    pub failed: usize,
    /// Cells explicitly skipped, with reasons.
    pub skipped: usize,
}

/// One cell, in artifact form: an executed tuple's verdict.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct CellRecord {
    /// Backend family, as `Debug` names it (`"Cpu"`, `"Cuda"`, ...).
    pub backend: String,
    /// Operation, as `Display` names it (`"add"`, `"matmul"`, ...).
    pub operation: String,
    /// Dtype, as the dtype registry names it (`"f32"`, `"q8_0"`, ...).
    pub dtype: String,
    /// Layout class, as `Debug` names it.
    pub layout: String,
    /// Rank this cell claims.
    pub rank: usize,
    /// Whether the row claims training-mode execution.
    pub training: bool,
    /// Math mode, as `Debug` names it.
    pub math_mode: String,
    /// Whether this cell is an unadvertised-dtype refusal probe.
    pub negative_probe: bool,
    /// `"passed"` or `"failed"`. Nothing else.
    pub status: &'static str,
    /// Why the cell failed. Absent on passes.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// One skipped tuple, as carried inside a [`SkippedGroup`].
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ExampleTuple {
    /// Dtype, as the dtype registry names it.
    pub dtype: String,
    /// Layout class, as `Debug` names it.
    pub layout: String,
    /// Rank the tuple claims.
    pub rank: usize,
    /// Whether the row claims training-mode execution.
    pub training: bool,
    /// Math mode, as `Debug` names it.
    pub math_mode: String,
}

/// Skipped tuples, grouped by `(backend, operation, skip)`.
///
/// Skip reasons repeat identically across hundreds of tuples: one coarse
/// operation's missing-fixture paragraph would otherwise be copied per cell.
/// The group records the reason once, the count it stands for, and one
/// example tuple so a reader can reproduce the skip.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct SkippedGroup {
    /// Backend family, as `Debug` names it.
    pub backend: String,
    /// Operation, as `Display` names it.
    pub operation: String,
    /// `"no-device"`, `"no-fixture"`, or `"unbuildable"`. Nothing else.
    pub skip: &'static str,
    /// How many tuples this group stands for.
    pub count: usize,
    /// Why the harness did not execute these tuples.
    pub reason: String,
    /// One tuple the group stands for.
    pub example: ExampleTuple,
}

fn skip_code(reason: &SkipReason) -> &'static str {
    match reason {
        SkipReason::NoDevice => "no-device",
        SkipReason::NoFixture(_) => "no-fixture",
        SkipReason::Unbuildable(_) => "unbuildable",
    }
}

fn record_of(cell: &MatrixCell) -> Option<CellRecord> {
    let (status, reason) = match &cell.verdict {
        CellVerdict::Passed => ("passed", None),
        CellVerdict::Failed(why) => ("failed", Some(why.clone())),
        // Skips are grouped in `skipped_groups`, not recorded per cell: the
        // reason text repeats identically across hundreds of tuples.
        CellVerdict::Skipped(_) => return None,
    };
    Some(CellRecord {
        backend: alloc::format!("{:?}", cell.backend),
        operation: cell.operation.to_string(),
        dtype: cell.dtype.clone(),
        layout: cell.layout.clone(),
        rank: cell.rank,
        training: cell.training,
        math_mode: cell.math_mode.clone(),
        negative_probe: cell.negative_probe,
        status,
        reason,
    })
}

/// Group key for skipped tuples: the reason text is identical within one.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SkipKey {
    backend: String,
    operation: String,
    skip: &'static str,
}

fn skipped_groups(report: &MatrixReport) -> Vec<SkippedGroup> {
    use std::collections::BTreeMap;

    let mut groups: BTreeMap<SkipKey, (usize, String, ExampleTuple)> = BTreeMap::new();
    for cell in &report.cells {
        let CellVerdict::Skipped(reason) = &cell.verdict else {
            continue;
        };
        let key = SkipKey {
            backend: alloc::format!("{:?}", cell.backend),
            operation: cell.operation.to_string(),
            skip: skip_code(reason),
        };
        groups
            .entry(key)
            .and_modify(|(count, _, _)| *count += 1)
            .or_insert((
                1,
                reason.reason(),
                ExampleTuple {
                    dtype: cell.dtype.clone(),
                    layout: cell.layout.clone(),
                    rank: cell.rank,
                    training: cell.training,
                    math_mode: cell.math_mode.clone(),
                },
            ));
    }
    groups
        .into_iter()
        .map(|(key, (count, reason, example))| SkippedGroup {
            backend: key.backend,
            operation: key.operation,
            skip: key.skip,
            count,
            reason,
            example,
        })
        .collect()
}

/// The whole artifact, in artifact form.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Artifact {
    /// [`SCHEMA_VERSION`]. A consumer that does not recognize it refuses.
    pub schema_version: u32,
    /// Per-leg counts, keyed by backend family name (`"Cpu"`, ...).
    pub summary: std::collections::BTreeMap<String, LegSummary>,
    /// One object per executed tuple: every pass and every failure.
    pub cells: Vec<CellRecord>,
    /// Skipped tuples, grouped by `(backend, operation, skip)`.
    pub skipped: Vec<SkippedGroup>,
}

fn leg_summary(report: &MatrixReport, backend: DeviceKind) -> LegSummary {
    LegSummary {
        advertised: report.leg(backend).count(),
        passed: report.passed_on(backend),
        failed: report.failed_on(backend),
        skipped: report.skipped_on(backend),
    }
}

impl MatrixReport {
    /// Render this report as the artifact the documentation generator reads.
    #[must_use]
    pub fn to_artifact(&self) -> Artifact {
        let mut summary = std::collections::BTreeMap::new();
        for backend in [
            DeviceKind::Cpu,
            DeviceKind::Cuda,
            DeviceKind::Wgpu,
            DeviceKind::Metal,
        ] {
            summary.insert(alloc::format!("{backend:?}"), leg_summary(self, backend));
        }
        let cells = self.cells.iter().filter_map(record_of).collect();
        Artifact {
            schema_version: SCHEMA_VERSION,
            summary,
            cells,
            skipped: skipped_groups(self),
        }
    }

    /// Serialize this report to compact JSON. Small by construction: one
    /// object per executed cell, grouped skips, no tensors, no values.
    pub fn to_json_string(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(&self.to_artifact())
    }

    /// Write this report to `path`, creating parent directories as needed.
    ///
    /// Returns the path written, so a test can report where the artifact went.
    pub fn write_json(&self, path: &std::path::Path) -> std::io::Result<String> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, self.to_json_string().map_err(std::io::Error::other)?)?;
        Ok(path.display().to_string())
    }
}
