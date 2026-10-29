## Summary

<!-- What does this PR do, and why? Link the issue it closes if there is one. -->

## Checklist

- [ ] `cargo fmt --all -- --check` passes
- [ ] Focused Clippy and tests were run for the affected crates per [`CONTRIBUTING.md`](../CONTRIBUTING.md); the CI CPU gates are `cargo clippy --all-targets --no-default-features --features incin-backends/cpu,incin/cpu -- -D warnings` and `cargo test --all-targets --no-default-features --features incin-backends/cpu,incin/cpu`
- [ ] Doctests were checked separately: `cargo test --workspace --doc --features incin-core/distributed` (`--all-targets` does not run doctests)
- [ ] `cargo check --workspace --all-features` passes (compile-only union check, not an all-feature lint or runtime test gate)
- [ ] Commands run and omitted hardware or broader validation are listed below
- [ ] Every new or moved `pub` item has a one-line summary and a runnable `# Examples` doctest, per [`docs/CONVENTIONS.md`](../docs/CONVENTIONS.md)
- [ ] A file that now mixes more than one concern was split by responsibility, not left to grow (`docs/CONVENTIONS.md`'s file-organization section); `tools/check-large-files.sh` passes
- [ ] `docs/book/src/` was updated if this PR changes user-facing behavior, and links to rustdoc rather than restating it
- [ ] `CHANGELOG.md` was updated if this PR is user-facing
- [ ] Public API surface changes are intentional: `bash tools/check-public-api.sh` passes, and the baseline diff (if any) was reviewed deliberately

## What changed and why

<!-- The design decision, not just the diff. If this touches something listed in
docs/FROZEN_FOUNDATIONS.md, say so explicitly and why it was still worth doing. -->

## How it was tested

<!-- Commands you ran, not just "tests pass". Include anything not covered by CI
(a manual run of an example, a hardware backend, a large workspace-wide check). -->
