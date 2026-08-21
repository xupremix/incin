---
name: Refactor / chore
about: Internal cleanup - a file split, a doc-comment pass, book alignment, or
  other work tracked against docs/CONVENTIONS.md rather than a bug or a new API
title: ""
labels: type:maintenance
assignees: ""
---

## What

<!-- The specific file, module, or doc chapter this covers. One item per issue
keeps the PR reviewable; if this is a batch (e.g. "split the remaining
cpu/ops/*.rs files"), list them explicitly. -->

## Why

<!-- Link the relevant contract, convention, release gate, or tracking issue. -->

## Done means

- [ ] No behavior or public API change, unless explicitly called out and justified
- [ ] The smallest relevant tests and lints pass
- [ ] Broader checks are listed when this crosses a crate or release boundary
- [ ] Every CI gate this touches stays in sync (`tools/check-large-files.sh`, `tools/check-hidden-items.py`, `docs/FROZEN_FOUNDATIONS.md`, `bash tools/check-public-api.sh`, as applicable)
