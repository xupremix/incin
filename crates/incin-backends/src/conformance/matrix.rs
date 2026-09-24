//! The registry-enumerated conformance matrix (issue #83).
//!
//! The execution half of this harness runs what the CPU registry advertises.
//! The matrix is the same enumeration widened to every backend family: one
//! row per advertised tuple per backend, each carrying an explicit verdict.
//! Adding a capability row automatically extends coverage, because the rows
//! are read off the backends' own registries rather than a hand list, and a
//! row cannot be advertised without being posed.
//!
//! # Verdicts, and what a skip means
//!
//! Every cell ends in exactly one of three states, and there is no fourth:
//!
//! * [`CellVerdict::Passed`] — the tuple executed (CPU), or the negative
//!   probe was refused (an unadvertised dtype the executor turned away).
//! * [`CellVerdict::Failed`] — an advertised tuple that refused, rejected,
//!   errored, panicked, recorded no training recipe, recorded one output
//!   twice, or an unadvertised tuple that executed. Any of these is a finding
//!   against the backend.
//! * [`CellVerdict::Skipped`] — the harness did not execute the tuple, with
//!   the reason recorded beside it. A skip is never silent: [`SkipReason`]
//!   says whether there is no device on this machine, no fixture yet, or the
//!   tuple could not be materialized.
//!
//! # Device legs without devices
//!
//! The CUDA, WGPU, and Metal legs enumerate their registries but do not
//! execute without a device on this machine: every cell reports
//! [`SkipReason::NoDevice`]. Running the same kernels against the wrong
//! backend's registry would test nothing, and executing device kernels with
//! no device is not a refusal the backend made. When the #82 runner lands,
//! the leg that gains a device replaces its skips with executions; the
//! enumeration is already the contract both will read.

use alloc::string::{String, ToString};
use alloc::vec::Vec;

use incin_core::shapes::error::OperationKind;
use incin_core::tensor::device::DeviceKind;

use crate::conformance::plan::{AdvertisedTuple, advertised_tuples};
use crate::conformance::{Coverage, Verdict, run_cpu_self_check};

/// Why the harness did not execute a tuple.
///
/// Kept distinct from a failure throughout. A harness that cannot build an
/// operand and reports that as a backend defect spends a reader's attention
/// on its own gaps, and a leg with no device that reports its tuples as
/// passed spends nobody's: it lies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    /// The backend family has no executable device on this machine. The
    /// tuple was enumerated from the registry and posed nowhere.
    NoDevice,
    /// No fixture for this operation yet, with the reason it is outstanding.
    NoFixture(&'static str),
    /// A fixture exists but this particular tuple cannot be materialized.
    Unbuildable(String),
}

impl SkipReason {
    /// A stable one-line reason for reports and the JSON artifact.
    #[must_use]
    pub fn reason(&self) -> String {
        match self {
            Self::NoDevice => "no executable device for this backend on this machine".to_string(),
            Self::NoFixture(why) => alloc::format!("no fixture yet: {why}"),
            Self::Unbuildable(why) => alloc::format!("tuple could not be materialized: {why}"),
        }
    }
}

/// What running one matrix cell concluded: pass, fail, or an explicit skip.
///
/// There is no fourth state. A cell that was never executed and never
/// recorded why would be a silent skip, which is the failure mode this
/// matrix exists to prevent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CellVerdict {
    /// The tuple executed (CPU leg), or the negative probe was refused.
    Passed,
    /// The backend broke its advertisement, with why.
    Failed(String),
    /// The harness did not execute the tuple, with why.
    Skipped(SkipReason),
}

impl CellVerdict {
    /// Whether this verdict is a finding against the backend.
    #[must_use]
    pub const fn is_finding(&self) -> bool {
        matches!(self, Self::Failed(_))
    }

    /// Whether the harness executed anything for this cell.
    #[must_use]
    pub const fn was_executed(&self) -> bool {
        matches!(self, Self::Passed | Self::Failed(_))
    }
}

/// One advertised tuple on one backend leg, and what it concluded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixCell {
    /// The backend family whose registry advertised the tuple.
    pub backend: DeviceKind,
    /// The exact catalog operation the row describes.
    pub operation: OperationKind,
    /// The one dtype this cell claims, as named by the dtype registry.
    pub dtype: String,
    /// The one layout class this cell claims.
    pub layout: String,
    /// The rank this cell claims.
    pub rank: usize,
    /// Whether the row claims training-mode execution.
    pub training: bool,
    /// The one math mode this cell claims.
    pub math_mode: String,
    /// What posing the cell concluded.
    pub verdict: CellVerdict,
    /// Whether this cell is a negative probe: an unadvertised dtype posed
    /// past admission to check the executor refuses it. A passed probe means
    /// the refusal held; a failed one means the backend executed something it
    /// never advertised.
    pub negative_probe: bool,
}

impl MatrixCell {
    /// A stable one-line identity for reports and failure messages.
    #[must_use]
    pub fn label(&self) -> String {
        alloc::format!(
            "{:?} {} [{}, {}, rank {}{}]",
            self.backend,
            self.operation,
            self.dtype,
            self.layout,
            self.rank,
            if self.negative_probe {
                ", negative probe"
            } else {
                ""
            },
        )
    }
}

/// Map the execution verdict onto the matrix verdict.
///
/// The refusal semantics live here, in one function, so they are unit-testable
/// without running the matrix:
///
/// * advertised-but-unexecutable fails (`Refused`, `Rejected`, `Failed`,
///   `Panicked`);
/// * `Training: yes` without a correct recording fails (`RecordedNothing`,
///   `RecordedOneOutputTwice`);
/// * unadvertised-but-executed fails (a negative probe that `Executed`
///   arrives here as `Failed` from the self-check, and stays failed);
/// * incompatible shapes and dtypes arrive as typed refusals, never panics:
///   a `Panicked` verdict is a finding, not a skip.
#[must_use]
pub fn map_oracle_verdict(verdict: &Verdict) -> CellVerdict {
    match verdict {
        Verdict::Executed => CellVerdict::Passed,
        Verdict::Refused(why)
        | Verdict::Rejected(why)
        | Verdict::Failed(why)
        | Verdict::Panicked(why) => CellVerdict::Failed(why.clone()),
        Verdict::RecordedNothing => CellVerdict::Failed(
            "the row claims training and the kernel recorded no tape node".to_string(),
        ),
        Verdict::RecordedOneOutputTwice(nodes) => CellVerdict::Failed(alloc::format!(
            "{nodes} tape nodes claim the same output identity; the reverse walk \
             doubles every gradient below this operation"
        )),
        Verdict::NotCovered(Coverage::Unfixtured(reason)) => {
            CellVerdict::Skipped(SkipReason::NoFixture(reason))
        }
        Verdict::NotCovered(Coverage::Unbuildable(reason)) => {
            CellVerdict::Skipped(SkipReason::Unbuildable(reason.clone()))
        }
    }
}

fn cpu_cell(tuple: &AdvertisedTuple, verdict: &Verdict, negative_probe: bool) -> MatrixCell {
    MatrixCell {
        backend: DeviceKind::Cpu,
        operation: tuple.operation,
        dtype: tuple.dtype.name().to_string(),
        layout: alloc::format!("{:?}", tuple.layout),
        rank: tuple.rank,
        training: tuple.training,
        math_mode: alloc::format!("{:?}", tuple.math_mode),
        verdict: map_oracle_verdict(verdict),
        negative_probe,
    }
}

fn skipped_cell(device: DeviceKind, tuple: &AdvertisedTuple) -> MatrixCell {
    MatrixCell {
        backend: device,
        operation: tuple.operation,
        dtype: tuple.dtype.name().to_string(),
        layout: alloc::format!("{:?}", tuple.layout),
        rank: tuple.rank,
        training: tuple.training,
        math_mode: alloc::format!("{:?}", tuple.math_mode),
        verdict: CellVerdict::Skipped(SkipReason::NoDevice),
        negative_probe: false,
    }
}

/// Everything one matrix run concluded, in enumeration order.
///
/// CPU cells carry real executions. The CUDA, WGPU, and Metal legs carry the
/// enumeration with [`SkipReason::NoDevice`] verdicts until a runner gives
/// them a device (issue #82). The order is the registries' table order, so a
/// report reads in the same sequence as the declaration a reader would check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MatrixReport {
    /// Every cell posed, CPU leg first, then one leg per device family.
    pub cells: Vec<MatrixCell>,
}

impl MatrixReport {
    /// Cells for one backend leg.
    pub fn leg(&self, backend: DeviceKind) -> impl Iterator<Item = &MatrixCell> {
        self.cells
            .iter()
            .filter(move |cell| cell.backend == backend)
    }

    /// Cells that are findings against their backend.
    pub fn failures(&self) -> impl Iterator<Item = &MatrixCell> {
        self.cells.iter().filter(|cell| cell.verdict.is_finding())
    }

    /// Whether nothing failed. Skips do not fail a report.
    #[must_use]
    pub fn passed(&self) -> bool {
        self.failures().next().is_none()
    }

    /// How many cells of one leg were actually executed.
    #[must_use]
    pub fn executed_on(&self, backend: DeviceKind) -> usize {
        self.leg(backend)
            .filter(|cell| cell.verdict.was_executed())
            .count()
    }

    /// How many cells of one leg passed.
    #[must_use]
    pub fn passed_on(&self, backend: DeviceKind) -> usize {
        self.leg(backend)
            .filter(|cell| cell.verdict == CellVerdict::Passed)
            .count()
    }

    /// How many cells of one leg failed.
    #[must_use]
    pub fn failed_on(&self, backend: DeviceKind) -> usize {
        self.leg(backend)
            .filter(|cell| cell.verdict.is_finding())
            .count()
    }

    /// How many cells of one leg were explicitly skipped.
    #[must_use]
    pub fn skipped_on(&self, backend: DeviceKind) -> usize {
        self.leg(backend)
            .filter(|cell| matches!(cell.verdict, CellVerdict::Skipped(_)))
            .count()
    }

    /// A one-line-per-leg summary plus every finding, for test output.
    #[must_use]
    pub fn summary_text(&self) -> String {
        let mut out = String::from("conformance matrix:\n");
        for backend in [
            DeviceKind::Cpu,
            DeviceKind::Cuda,
            DeviceKind::Wgpu,
            DeviceKind::Metal,
        ] {
            let total = self.leg(backend).count();
            out.push_str(&alloc::format!(
                "  {:?}: {} advertised, {} passed, {} failed, {} skipped\n",
                backend,
                total,
                self.passed_on(backend),
                self.failed_on(backend),
                self.skipped_on(backend),
            ));
        }
        for cell in self.failures() {
            let detail = match &cell.verdict {
                CellVerdict::Failed(why) => why.clone(),
                _ => String::new(),
            };
            out.push_str(&alloc::format!("  FAIL {}: {detail}\n", cell.label()));
        }
        out
    }
}

/// Run the full matrix: execute the CPU leg, enumerate the device legs.
///
/// The CPU leg reuses the oracle self-check, including its negative probes:
/// an unadvertised dtype posed past admission. A probe the executor refused
/// maps to [`CellVerdict::Passed`] (the refusal held); one the executor ran
/// maps to [`CellVerdict::Failed`] (unadvertised-but-executed).
///
/// Never panics and never stops early. A run is read once, and a reader wants
/// every finding from it rather than the first.
#[must_use]
pub fn run_matrix() -> MatrixReport {
    let mut cells = Vec::new();

    let oracle = run_cpu_self_check();
    // Negative probes share the observation list with advertised tuples. A
    // probe carries a dtype its operation's row does not advertise, which is
    // exactly what distinguishes it: no advertised tuple can disagree with
    // its own row.
    let registry = crate::capability::registry(DeviceKind::Cpu);
    let advertised = |tuple: &AdvertisedTuple| {
        registry
            .registrations()
            .iter()
            .find(|rule| rule.operation == tuple.operation)
            .is_some_and(|rule| {
                rule.dtypes
                    .iter()
                    .any(|dtype| dtype.key() == tuple.dtype.key())
            })
    };
    for observation in &oracle.observations {
        let negative_probe = !advertised(&observation.tuple);
        cells.push(cpu_cell(
            &observation.tuple,
            &observation.verdict,
            negative_probe,
        ));
    }

    for device in [DeviceKind::Cuda, DeviceKind::Wgpu, DeviceKind::Metal] {
        for tuple in advertised_tuples(device) {
            cells.push(skipped_cell(device, &tuple));
        }
    }

    MatrixReport { cells }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conformance::fixtures::Coverage;

    /// Advertised-but-unexecutable fails: every refusal-shaped verdict is a
    /// finding, never a skip and never a pass.
    #[test]
    fn refusal_shaped_verdicts_map_to_failure() {
        for verdict in [
            Verdict::Refused("the registry does not admit its own tuple".to_string()),
            Verdict::Rejected("the descriptor refused the invocation".to_string()),
            Verdict::Failed("the kernel errored".to_string()),
            Verdict::Panicked("panicked".to_string()),
        ] {
            assert!(
                map_oracle_verdict(&verdict).is_finding(),
                "{verdict:?} must be a finding against the backend"
            );
        }
    }

    /// `Training: yes` without a correct recording fails: a row that records
    /// nothing, or records one output twice, holds a hole in the graph either
    /// way, and the tape-depth check alone cannot separate the two.
    #[test]
    fn missing_or_doubled_recordings_map_to_failure() {
        assert!(map_oracle_verdict(&Verdict::RecordedNothing).is_finding());
        assert!(map_oracle_verdict(&Verdict::RecordedOneOutputTwice(2)).is_finding());
    }

    /// Unadvertised-but-executed fails: a negative probe the executor ran
    /// arrives as `Failed` from the self-check, and the mapping keeps it
    /// failed rather than reinterpreting it.
    #[test]
    fn an_executed_negative_probe_stays_a_failure() {
        let verdict = Verdict::Failed("executed f32, which the row does not advertise".to_string());
        assert!(map_oracle_verdict(&verdict).is_finding());
    }

    /// A refused negative probe passes: the contract held, and the matrix
    /// must not report a held contract as a defect.
    #[test]
    fn a_refused_probe_maps_to_passed() {
        assert_eq!(map_oracle_verdict(&Verdict::Executed), CellVerdict::Passed);
    }

    /// Harness gaps map to explicit skips with reasons, never to failures and
    /// never to passes: a backend must not be blamed for a fixture the
    /// harness has not written.
    #[test]
    fn harness_gaps_map_to_reasoned_skips() {
        let unfixtured = map_oracle_verdict(&Verdict::NotCovered(Coverage::Unfixtured("why")));
        assert_eq!(
            unfixtured,
            CellVerdict::Skipped(SkipReason::NoFixture("why"))
        );
        let unbuildable = map_oracle_verdict(&Verdict::NotCovered(Coverage::Unbuildable(
            "why".to_string(),
        )));
        assert_eq!(
            unbuildable,
            CellVerdict::Skipped(SkipReason::Unbuildable("why".to_string()))
        );
        for verdict in [unfixtured, unbuildable] {
            assert!(!verdict.is_finding());
            assert!(!verdict.was_executed());
            let CellVerdict::Skipped(reason) = verdict else {
                unreachable!("matched Skipped above");
            };
            assert!(!reason.reason().is_empty());
        }
    }
}
