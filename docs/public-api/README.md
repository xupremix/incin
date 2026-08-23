# Reviewed public API baselines

Every shipped crate's public API is snapshotted with `cargo-public-api`
(`-sss`: blanket, auto-trait, and auto-derived implementations omitted)
and reviewed as a checked-in baseline under this directory:

| Baseline | Package | Feature set |
| --- | --- | --- |
| `incin-cpu.txt` | `incin` | default CPU facade |
| `incin-core-std.txt` | `incin-core` | `std` |
| `incin-backends-cpu.txt` | `incin-backends` | `std,cpu` |
| `incin-data.txt` | `incin-data` | defaults |
| `incin-diagnostics.txt` | `incin-diagnostics` | defaults |
| `incin-telemetry.txt` | `incin-telemetry` | defaults |
| `incin-macros.txt` | `incin-macros` | n/a (proc-macro surface) |
| `incin-lsp.txt` | `incin-lsp` | defaults |
| `incin-viz-plugin-api.txt` | `incin-viz-plugin-api` | defaults |
| `incin-viz.txt` | `incin-viz` | defaults |

Run `python3 tools/check-public-api-baseline.py` after changing any
shipped crate's public surface. An addition, removal, or signature
change fails the check until the change is reviewed and the affected
baseline is regenerated in the same commit
(`tools/check-public-api-baseline.py --update <crate>`); a single crate
can be checked alone by passing its package name.

Preview surfaces are excluded from these baselines by feature selection,
not by post-filtering: `incin::experimental::compiled` and the other
feature-gated namespaces (see `docs/PROJECT_STATUS.md` for the preview
list) never compile into the baseline feature sets above. Their contracts
are tracked by their own preview fixtures instead.

SemVer enforcement against these baselines runs in CI through
`cargo-semver-checks`, diffing each push against the most recent release
tag; until the first tag exists it records a skip rather than a pass.

The baselines intentionally cover shipped packages only.
`tools/check-package.sh` independently verifies archive contents for
exactly that set, so an API baseline cannot stand in for packaging
validation or vice versa.
