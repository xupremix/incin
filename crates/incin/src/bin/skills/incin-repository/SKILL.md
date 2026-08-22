---
name: incin-repository
description: Use when working anywhere in the Incin repository, including navigation, documentation, the book or Pages site, CI, releases, crates.io packaging, GitHub issues/settings, cargo-incin, incin-lsp, or editor integrations. Route substantive Rust implementation and contract-sensitive design to incin-engineering.
---

# Incin repository workflow

Use this skill for repository-wide orientation and non-Rust work. It complements
`incin-engineering`: use that skill for substantive Rust implementation,
backend/core/macro changes, public APIs, invariants, performance, and debugging.
Do not duplicate its implementation or testing contract here.

## Start with repository evidence

Read `AGENTS.md` first. If `graphify-out/graph.json` exists, immediately run a
narrow query before broad browsing:

```text
graphify query "<the concrete question>"
```

Use `graphify path` or `graphify explain` for relationships and focused
concepts. Then use `rg`/`rg --files` to locate the exact file and read the
smallest relevant section. After code changes, run `graphify update .` when
available; derived graph output is not hand-maintained.

## Know which files are authoritative

Before editing documentation, read `docs/README.md`.

- Binding architecture and API contracts: `docs/FROZEN_FOUNDATIONS.md`,
  `docs/API_DESIGN.md`, `docs/ERROR_CONTRACT.md`, and `docs/INVARIANT_TYPES.md`.
- User guidance: `docs/GUIDE.md` and `docs/book/src/`.
- Status and plans: `docs/PROJECT_STATUS.md`, `docs/plan/`, and `docs/growth/`;
  verify important claims against source, tests, or audit evidence.
- Generated documents: `docs/capabilities.md`,
  `docs/OPERATION_SEMANTICS.md`, `docs/operation-coverage.md`, shape audits,
  and migration evidence. Never hand-edit them; run the generator named in
  `docs/README.md`.
- Derived directories such as `graphify-out/`, `target/`, and generated site
  output are not source of truth.

## Route common work to its source

- Workspace/crate metadata: root `Cargo.toml`, each crate manifest, and
  `cargo metadata --no-deps --format-version 1`.
- `cargo incin`: `crates/incin/src/bin/cargo-incin.rs` and its facade crate.
- LSP proxy: `crates/incin-lsp/`; its editor clients live under
  `editors/vscode/` and `editors/nvim/`.
- RustRover fallback: `editors/rustrover/`.
- CI contracts: `.github/workflows/` and the checks they invoke; inspect the
  workflow before changing a release or Pages assumption.
- Release integrity and artifact contract: `docs/RELEASE.md`. Release operator
  procedure: `docs/RELEASING.md`. Use both with `tools/release-preflight.py`,
  `tools/release-assets.py`, and `.github/workflows/release.yml`.
- Book and Pages: `docs/book/src/SUMMARY.md`, `docs/book/`,
  `docs/book/build_site.py`, `.github/workflows/pages.yml`, and the book/site
  check scripts. Keep navigation and generated output consistent.
- Issues, labels, and security settings: repository GitHub metadata and the
  relevant workflow/configuration. Keep issue text scoped to one verifiable
  outcome and avoid duplicating an existing issue.

For a Rust change, hand the discovered files and contract to
`incin-engineering`; do not invent a public API, feature flag, invariant, or
release compatibility policy in this skill.

## Validate proportionally

Run the smallest check matching the touched surface, then broaden only when a
boundary is crossed:

- Markdown/config/scripts: the local checker named beside the file.
- Book: `mdbook build docs/book`, `python3 tools/check-book-site.py`, and the
  documented browser test when routing/theme behavior changed.
- VS Code: `npm ci`, `npm run compile`, and the package/test command in
  `editors/vscode/README.md`.
- Neovim: the focused Lua tests or headless smoke test documented in
  `editors/nvim/README.md`.
- LSP and `cargo incin`: focused crate tests/builds, then a local binary smoke
  test when packaging or editor launch behavior changed.
- Release: run the repository preflight and artifact verification before any
  tag or publication. Do not use the full CI fan-out as a first check.

Report changed files, checks and results, intentionally omitted broader checks,
and remaining risks.

## Branches and external mutations

Inspect `git status`, branch tracking, remotes, and workflow triggers before
promoting work. Preserve unrelated changes. Treat pushes, merges to the
default branch, GitHub settings, issue/label creation, registry publication,
Marketplace uploads, and release publication as external mutations.

Before a release, establish one version across workspace manifests, the tag,
book/editor metadata, and generated release assets. Use draft releases and
dry-run/preflight checks first. Publishing to crates.io or an editor
marketplace is irreversible or difficult to undo; never publish merely because
the code compiles, and stop when credentials, ownership, or a release decision
is missing. Keep `incin-lsp` binaries and editor packages aligned with the
release workflow rather than assuming a crates.io package or marketplace
listing exists.

When capturing editor screenshots, use an isolated demo workspace and local
configuration. Inspect images and metadata before committing them; remove
usernames, home paths, tokens, machine identifiers, logs, and other incidental
information. Do not claim an editor or release path is verified without a real
smoke test and recorded evidence.
