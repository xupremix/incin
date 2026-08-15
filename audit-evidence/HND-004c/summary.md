# HND-004c validation evidence

Start commit: `2db03c1`.

HND-004c commits:

- `7fbf8cc fix: make StatePath construction fallible`
- `7f1ebb6 docs: align current regression references`
- `1e98af4 test: align macro coverage with current APIs`

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

## Resource-bounded validation

The exact CI core powerset enumerates 384 combinations. It was started and
reached combination 9 without a failure, then stopped after the observed rate
made completion impractical in this environment. The unbounded facade probe
enumerated 279,552 combinations before being stopped during combination 3.
Backend and facade final matrices are therefore `UNAVAILABLE`, not passed.
The interrupted logs are retained beside this file. No feature-matrix pass is
claimed for those incomplete runs.

## Audit conclusions

The retired capability and gradient integration files were audited from Git
history. Their useful current coverage exists in capability registration tests,
backend unit tests, core autograd tests, and current canonical dispatch tests.
Current CI and ledger references no longer name the deleted test binaries.

The canonical export is produced only after the final documentation commit
with `tools/export-snapshot.sh`; its size and SHA-256 are recorded in the final
handoff section and in the export log.
