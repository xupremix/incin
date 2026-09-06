# Compatibility and deprecation policy

This document states what consumers may rely on at each maturity stage, how
deprecations are announced, and how fixes reach shipped releases. It is the
policy `cargo-semver-checks` and the public API baselines enforce against.

## Versioning scheme

Incin follows Semantic Versioning: `MAJOR.MINOR.PATCH`.

- **MAJOR** changes break documented compatibility.
- **MINOR** releases add features and may change behavior where the previous
  behavior was documented as unspecified; deprecations (post-1.0) land here.
- **PATCH** releases contain bug fixes and security fixes with no intended
  API or behavioral change beyond the fix itself.

## Pre-1.0 (current: 0.x)

Before 1.0, the API is intentionally unstable while foundations settle:

- The reviewed baselines under `docs/public-api/` are frozen per release, so
  *accidental* surface drift still fails CI - but a deliberate breaking
  change between 0.x versions is allowed when recorded in the CHANGELOG and,
  where it touches a frozen foundation, in the decision ledger
  (`PROPOSALS.md :: Appendix C`).
- Preview namespaces (`incin::experimental::*`, preview Cargo features) carry
  no compatibility promise at any version; they are excluded from baselines
  by feature selection.
- Pin exact versions (`=0.1.0`) if you need stability during this phase.

## Post-1.0

- Within a major version, existing documented behavior does not break.
- **Deprecations** are marked with `#[deprecated]` (or documented equivalent)
  and announced in the CHANGELOG one minor release before removal. Removal
  happens no earlier than the next major release.
- New public surface requires a reviewed baseline update in the same commit
  that introduces it.

## Release trains and backports

- Development happens on `master`; it must remain green.
- A stabilization branch `release/X.Y` is cut at each release candidate.
  Final X.Y.0 tags come from that branch.
- Fixes for reported bugs in a shipped release land on `master` first, then
  are cherry-picked to the active `release/X.Y` branch through a PR labeled
  `backport`. Patch tags (`X.Y.Z`) cut from the release branch after review.
- Security fixes follow the same path with the advisory timeline from
  [SECURITY.md](../SECURITY.md) taking precedence over regular train cadence.

## What is exempt

Generated documents (`docs/capabilities.md`, operation semantics), audit
evidence, and derived directories are outputs of the source and tests; their
content may change in any release as truth changes, without notice.

## Hardware and feature-combination coverage

Compiling every supported feature combination is a per-PR gate (the
`all-features-check` union job and the feature-contract matrix in CI);
executing them is not. The CUDA, Metal, native-WGPU, multi-host NCCL, and
other device suites need hardware no per-PR runner has, so they run on the
scheduled hardware matrix (`.github/workflows/hardware.yml`), which states
what it skipped and why when no runner is registered. Per-PR, the
accelerator backends get compile coverage of all their targets (library,
tests, examples) without hardware. A green PR therefore means "every
supported combination compiles and the CPU surface passes", never "the
device surface was executed".
