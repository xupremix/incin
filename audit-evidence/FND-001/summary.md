# FND-001 Evidence Summary

Status: DONE

Starting commit: `21ba8903d6e550f509a9e85639f3216821fb1265`.

## Contract established

The stable root and default prelude are explicit end-user allow-lists. External backend contracts are available only through the feature-gated `incin::backend_authoring` tier. Compiler/import/training/tuning/distributed surfaces are under `incin::experimental`. `DummyBackend` is available only through feature-gated `incin::test_utils`.

The audited cross-crate wildcard query returns no matches. Internal module-local wildcards inside owning crates remain an organization detail and are not forwarded by the `incin` facade.

## Acceptance gate

| Criterion | Status | Evidence |
|---|---|---|
| Explicit stable root and prelude | PASS | `public-api-before.txt`, `public-api-after.txt`, and `public-api.diff` |
| Backend-authoring, experimental, and test tiers | PASS | `check-tier-matrix.txt` and `facade-contract.txt` |
| `Dyn` usable as a type and value marker | PASS | isolated `default-pass` and `no-default-pass` consumer fixtures |
| Internal and disabled names absent | PASS | isolated compile-fail consumer fixtures and reviewed diagnostics |
| Default/no-default/std/CPU feature checks | PASS | `check-*.txt` |
| Workspace Clippy, tests, doctests, and rustdoc | PASS | successful `*-rerun.txt`, package results, `doctest-workspace.txt`, and `rustdoc-workspace.txt` |
| Public migration reviewed | PASS | `migration-table.md` |
| Graphify synchronized after source changes | PASS | `graphify-update.txt` |
| `cargo-semver-checks` for `incin` | BLOCKED | `semver-incin.txt`: the tool forces every feature and fails while rustdoc builds the pre-existing incomplete Candle/accelerator combination |

The initial archived Clippy/workspace/`incin`/macro runs identified moved-path integration failures. These were repaired with explicit imports and updated trybuild fixtures; the corresponding reruns exit 0. Both the initial and final outputs are retained.

`cargo fmt --all -- --check` retains the previously archived workspace-wide formatting drift outside this task. `fmt-focused.txt` and `diff-check.txt` prove the FND-001 Rust files and diff are clean.
