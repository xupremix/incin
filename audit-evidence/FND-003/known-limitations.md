# Known limitations

- The repository-wide formatter baseline remains non-clean. Its exact failure
  is archived in `test-results/fmt-workspace-final.txt`; the task-local Rust
  files and diff hygiene pass.
- `cargo semver-checks` could not produce a comparison. Its forced all-feature
  scratch build resolved a newer Candle dependency whose expanded dtype enum
  is not covered by the locked-workspace adapter. The exact failure is archived
  in `test-results/semver-incin-final.txt`.
- CUDA and Metal hardware execution was not run. Their feature builds pass;
  this task makes no hardware-behavior claim. WGPU tests passed with the
  available adapter, but that does not establish portability across adapters.
- Legacy free-form `Error::Msg`, `Error::BackendFailure`, and compatibility
  variants remain. New foundation paths use the typed bounded contract; a
  workspace-wide removal is intentionally not claimed.
- The legacy `StateDict::state_dict` export signature is infallible. Validated
  optimizer state is representable, but fallible serialization is outside
  FND-003.
- Tensor operator outputs are intentionally source-breaking `Result` values;
  no panic-preserving compatibility shim exists.
- Canonical operation identities, exact per-operation semantics, capability
  generation, and descriptor coverage are deferred to FND-004. The legacy
  operation-family execution architecture remains until FND-005.
- Graphify updated the AST graph but reported stale community labels and ten
  fail-closed nodes retained from files outside the scan corpus.
