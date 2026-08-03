# Known limitations

- The repository-wide formatter baseline remains non-clean. Its exact failure
  is archived in `test-results/fmt-workspace.txt`; the task-local Rust files and
  diff hygiene pass independently.
- CUDA and Metal execution were not run on physical devices. Their feature
  builds pass, but this task makes no hardware-behavior claim. WGPU physical
  adapter availability is likewise not inferred from selector construction.
- The Graphify AST update completed, but Graphify reported stale community
  labels and retained ten fail-closed nodes whose files left the scan corpus.
  No semantic graph claim relies on those nodes.
- FND-002 does not establish the FND-003 typed-error, checked scalar conversion,
  or optimizer rollback contracts. Existing float-to-integer behavior is
  therefore not claimed as remediated here.
- FND-002 does not change the legacy operation-family execution architecture;
  that remains gated on FND-004 and FND-005.

