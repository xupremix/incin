# Current feature-matrix status

Final source commit under validation: `93d135d` plus the pending evidence
update below.

## Completed

- Core: exact CI command
  `cargo hack check -p incin-core --feature-powerset --all-targets
  --exclude-features nightly`: 384/384 combinations completed successfully.
- Macros: 12 combinations passed, archived in `powerset-macros-final.log`.
- Diagnostics: 3 combinations passed, archived in `powerset-small.log`.
- Soundness: corrected backend Miri completed 322 tests with 3 ignored
  numerical tests; baseline and AVX2 ASan passed. Logs are archived as
  `miri-backend-final.log` and `asan-final.log`.

## Incomplete and not claimed as passed

- Backend: the exact current CI command expands to 8,212 generated commands.
  The generated command list is archived as
  `powerset-backends-command-list-current.txt`. A shared-target partitioned
  attempt was stopped during initial compilation after resource and wall-time
  inspection; the partial logs record no compiler error, but the run is not a
  pass.
- Facade: the exact 36,608-combination command was not run after the resource
  assessment. No facade powerset pass is claimed.

The workspace, CPU, docs, clippy, and structural validation results are in
`final-matrix.log`, `final-checks-current.log`, and
`final-workspace-current.log`.
