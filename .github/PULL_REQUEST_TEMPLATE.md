## Summary

<!-- What does this PR do, and why? Link the issue it closes if there is one. -->

## Checklist

- [ ] `cargo fmt --all -- --check` passes
- [ ] `cargo clippy --workspace --all-targets --all-features -- -D warnings` passes
- [ ] `cargo test --workspace --all-features` passes, including doctests
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
