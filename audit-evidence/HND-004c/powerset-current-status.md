# Current feature-matrix status

Final source commit under validation: pending the evidence commit below.

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

- Backend: exact CI command began, but sequential execution was stopped at
  49/8212 after storage and wall-time inspection. An equivalent fixed-feature
  partition attempt was also stopped before completion because separate Cargo
  targets duplicated artifacts. The partial logs record no source compile
  failure; their interrupted commands exited 130 or 1 from the stop.
- Facade: the exact 36,608-combination command was not run after the resource
  assessment. No facade powerset pass is claimed.

The workspace, CPU, docs, clippy, and structural validation results are in
`final-matrix.log`.
