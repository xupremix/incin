# FND-000 Known Limitations

- The required supporting file `docs/FOUNDATION_REVIEW_AND_EXECUTION_PLAN.md`
  was absent from both the checkout and inspected Git history.
- The starting workspace fails `cargo fmt --all -- --check` because of broad
  pre-existing formatting drift. Unrelated files are not reformatted in this
  task.
- The initial strict workspace Clippy run exposed current-toolchain lints. The
  bounded findings were corrected and the final strict Clippy run exits 0.
- The starting workspace test run failed on two `trybuild` snapshots whose
  trait paths changed under the current compiler. The snapshots were reviewed,
  the final workspace run exits 0, and this does not retroactively verify the
  historical test count.
- Graphify was available but stale at task start, so source inspection is the
  authority for reproduced findings.
- ONNX initializer loading and control flow remain intentionally unsupported.
- Compiled execution remains a structural prototype behind an experimental,
  opt-in boundary.
