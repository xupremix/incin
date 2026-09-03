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

### Tests that need a device

A test which cannot run everywhere is marked `#[ignore]` with a reason string,
and it **fails** when the device is absent rather than returning early. Call
`require_cuda()` or `require_wgpu()`; do not write

```rust,ignore
if !device_available() { return; }   // reports `ok` for a test that ran nothing
```

A skipped test and a passing test produce the same green line, in the log, the
summary and the badge. Three CUDA defects survived release-to-release behind
that pattern — optimizers that never launched, an `argmax` that computed one
row, and an embedding module that did not compile — because the suites covering
them were reporting success without running.

The same applies to swallowing a failure: `Err(_) => return` inside a test hides
exactly the breakage the test exists to find.

`cargo xtask hardware-tests` derives the expected hardware-test count from the
`#[ignore]` reasons in the tree, so a new reason string must be classified in
`xtask/src/hardware.rs` as either running on the CUDA runner or deliberately
excluded. An unrecognised reason fails the gate on purpose.

## Code of Conduct

Participation in the Incin project is governed by the [Code of Conduct](CODE_OF_CONDUCT.md)
(Contributor Covenant v2.1). Be respectful, keep technical disagreement about
the work, and do not harass or demean other contributors. Incidents may be
reported to **xupremix.me@gmail.com**.

## Releases

Release tags build and upload a draft containing the Book, editor integrations,
`incin-lsp`, and `cargo-incin`. Publishing that draft is a separate protected
action. See [docs/RELEASING.md](docs/RELEASING.md) for the full procedure.

## Engineering workflow

The branching and release model mirrors mainstream OSS projects (PyTorch,
TensorFlow):

- **Trunk-based.** `master` is the integration branch and must stay green.
  Work happens on short-lived feature branches named for their issue
  (`fix/issue-43-typed-errors`, `feat/issue-18-api-freeze`). Delete branches
  once their PR merges.
- **Pull requests only.** Direct pushes to `master` are not part of the
  workflow. Every change lands through a PR that closes a tracked issue,
  with CI green before merge.
- **PR titles** follow Conventional Commits (`feat:`, `fix:`, `docs:`,
  `chore:`, `test:`), optionally scoped - `fix(data): ...`. This matches the
  commit style in history and keeps the CHANGELOG diffable.
- **Release stabilization** uses `release/X.Y` branches cut at the release
  candidate; final tags come from the branch, not from master.
- **Backports**: fixes land on master first, then are cherry-picked to the
  active release branch through a PR labeled `backport`. Patch versions
  (`X.Y.Z`) tag from the release branch. See
  [docs/COMPATIBILITY.md](docs/COMPATIBILITY.md) for the full policy.
- **Deprecations** (post-1.0) are announced one minor release before
  removal; pre-1.0 carries no cross-version guarantee (pin exact versions).
