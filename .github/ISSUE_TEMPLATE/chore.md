---
name: Refactor / chore
about: Internal cleanup - a file split, a doc-comment pass, book alignment, or
  other work tracked against docs/CONVENTIONS.md rather than a bug or a new API
title: ""
labels: chore
assignees: ""
---

## What

<!-- The specific file, module, or doc chapter this covers. One item per issue
keeps the PR reviewable; if this is a batch (e.g. "split the remaining
cpu/ops/*.rs files"), list them explicitly. -->

## Why

<!-- Which convention in docs/CONVENTIONS.md this addresses (file organization,
doc-comment shape, examples/tests, or book alignment), or link the tracking
entry in PROPOSALS.md / .superpowers/sdd/large-file-splits/progress.md if
this is part of that effort. -->

## Done means

- [ ] No behavior or public API change, unless explicitly called out and justified
- [ ] `cargo test --workspace --all-features` and `cargo clippy --workspace --all-targets --all-features -- -D warnings` still pass
- [ ] Every CI gate this touches stays in sync (`tools/check-large-files.sh`, `tools/check-hidden-items.py`, `docs/FROZEN_FOUNDATIONS.md`, `bash tools/check-public-api.sh`, as applicable)
