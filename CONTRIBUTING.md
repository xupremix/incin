# Contributing to Incin

Search the issue tracker before opening a bug or proposal. Keep each issue and
pull request focused on one verifiable outcome.

## Issue triage

Maintainers apply four label groups:

- one `type:*` label for the kind of work;
- one or more `area:*` labels for ownership;
- one `priority:*` label for ordering;
- either `release:0.1` or `release:post-0.1` for release scope.

`status:*` and `breaking-change` add constraints when needed. GitHub's default
`bug`, `enhancement`, and `documentation` labels remain available for
compatibility with existing links and searches.

## Development setup

1. Install the Rust toolchain named in `rust-toolchain.toml`.
2. Run the smallest build or test for the crate you plan to change.
3. Read `docs/README.md` before editing documentation or generated files.
4. Read the relevant binding contract before changing a public API, error,
   invariant-carrying type, or frozen foundation.

Hardware and preview backends are opt-in. Do not enable every feature as a
generic first check; use the focused command documented for the surface you
changed.

## Submitting a pull request

- Run `cargo fmt --check` and the focused Clippy and test commands for the
  affected crates.
- Add regression or negative coverage for behavior changes.
- Document intentional public items and update migration notes for contract
  changes.
- Update the Book, generated checks, or release notes when user-visible
  behavior changes.
- Link the issue the pull request resolves and list any hardware or broad
  validation that was not run.

## Code of Conduct

Participation in the Incin project is governed by the [Code of Conduct](CODE_OF_CONDUCT.md)
(Contributor Covenant v2.1). Be respectful, keep technical disagreement about
the work, and do not harass or demean other contributors. Incidents may be
reported to **xupremix.me@gmail.com**.

## Releases

Release tags build and upload a draft containing the Book, editor integrations,
`incin-lsp`, and `cargo-incin`. Publishing that draft is a separate protected
action. See [docs/RELEASING.md](docs/RELEASING.md) for the full procedure.
