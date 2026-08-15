# HND-004c validation evidence

Start commit: `2db03c1`.

HND-004c commits:

- `7fbf8cc fix: make StatePath construction fallible`
- `7f1ebb6 docs: align current regression references`
- `1e98af4 test: align macro coverage with current APIs`
- `93d135d docs: align current defaults and regression evidence`

Environment: Linux x86_64, `rustc 1.97.1 (8bab26f4f 2026-07-14)`,
`cargo 1.97.1 (c980f4866 2026-06-30)`, `cargo-hack 0.6.45`.

## Passed

- StatePath tests: 9 passed. Root uses the empty wire string, nested paths
  round-trip, invalid components are rejected, and public child construction
  is fallible through `try_child`.
- Macro compile-pass and compile-fail suite: 4 passed.
- Current capability registration tests: 5 passed.
- Current CPU tape tests: 10 passed.
- Transformer proof: 2 passed.
- Distributed macro regression: 1 passed.
- Generated capability docs: 4 passed.
- `cargo fmt --all --check`, `cargo xtask ledger`, `cargo xtask budgets`,
  `tools/audit-shapes.sh --check`, architecture, dependency, large-file, and
  public-API gates all exited zero.
- Macro feature powerset: 12 combinations passed.
- Diagnostics feature powerset: 3 combinations passed.
- Compile benchmark with `CLEAN_EACH=1`: every case passed. The final
  incremental portfolio is reported separately and is not a clean sample.
- Final workspace matrix: formatting, workspace check, workspace tests,
  workspace doctests, workspace clippy with `-D warnings`, crate checks/tests,
  workspace docs, mdBook, architecture, dependency, large-file, public-API,
  and package gates all exited zero. Full output is in `final-matrix.log`.
- Soundness constituents passed: corrected backend Miri completed 322 tests
  with 3 ignored numerical tests, and baseline plus AVX2 ASan completed
  successfully. The backend Miri log is `miri-backend-final.log`; the ASan
  log is `asan-final.log`.

## Resource-bounded validation

The exact CI core powerset completed all 384 combinations successfully; its
complete log is `powerset-core-final.log`. The current backend command expands
to 8,212 generated `cargo check` commands. The exact command list is archived
as `powerset-backends-command-list-current.txt`; a shared-target partitioned
run and a separate-target retry were stopped during initial compilation after
resource and wall-time inspection, with no compiler error observed in the
saved output. Their partial logs and the retry command list are archived
beside this file. This is incomplete validation, not a backend powerset pass.
The facade command reports 36,608 combinations and was not run after the same
assessment. Backend and facade final matrices are therefore unavailable, not
passed.

The current bounded final checks are archived in `final-checks-current.log`
and `final-workspace-current.log`; both recorded exit status zero. Earlier
soundness constituent logs remain `miri-backend-final.log` and `asan-final.log`.

## Audit conclusions

The retired capability and gradient integration files were audited from Git
history. Their useful current coverage exists in capability registration tests,
backend unit tests, core autograd tests, and current canonical dispatch tests.
Current CI and ledger references no longer name the deleted test binaries.

The canonical export will be regenerated after the final evidence and
documentation commit; its size and SHA-256 are reported with the final
handoff.
