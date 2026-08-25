# Release evidence: 0.1.0 candidate

Branch: `integration/0.1.0`
Release base: `094bb484`
Candidate source commit: `27d874fd`
Candidate tree: `c43b28cc4f0878b07e324f98c6ee97e62d0fd210`

The aggregate gate ran against `27d874fd`. The commit carrying this bundle sits
on top of it and adds only files under `audit-evidence/REL-0.1.0-final/`, which
`git show --stat HEAD` confirms, so the branch tip has a different tree hash
than the one recorded above and that is expected.

This bundle covers the combined repository at the commit named above, not a
single line of work. Two streams of work were developed in parallel from the
same base and merged here with `--no-ff` so each retains its own ancestry: an
architecture, public API, release, and documentation stream, and an
implementation correctness stream. Every validation result recorded in
[`validation.txt`](validation.txt) was produced against the candidate tree.

## Commits in the candidate

Public API, release, and documentation:

`ad716155` demote kernel iteration plumbing out of the 0.1 surface
`2a199a2b` record the API tier classification in the changelog
`c0754c02` explain the err and error split in the tier table
`53a084d6` apply rustfmt to the merged reduce gradient work
`69eeb434` let the inspect smoke check run on a fresh clone

Implementation correctness:

`0031816a` record batch-norm training reductions on the autograd tape
`a3dcbd79` give cumsum and prod the gradients their catalog rows promise
`e7001dc0` pin zero-sized operand semantics to the catalog's EmptyRule
`e21bcba7` allow the type-complexity lint in the zero-sized suite
`6c2a1ee4` gradcheck reduction backwards over non-contiguous operands

Integration points: `be21ea50`, `f1b54c65`, `afc316fb`, `27d874fd`.

## The three briefed release blockers were already closed

Each was verified against the tree rather than taken from audit history:

| Briefed blocker | Actual state at `094bb484` |
| --- | --- |
| Missing `docs/assets/editors/*.png` | Both present, real 1440x900 PNGs, not placeholders |
| Stale hidden-item entry for `crates/incin-backends/src/target/ext.rs` | The file exists; the entry is live, not stale |
| `cargo fmt --all -- --check` | Passed |

One gap in that conclusion was worth closing. `tools/check-hidden-items.py`
walks source files and asserts each hidden item appears in the inventory, so it
detects a *missing* row but never an *orphan* one: a row pointing at a deleted
file passes silently. Every `crates/**.rs` path referenced by
`docs/public-api/hidden-items.md` was therefore checked for existence. 22
paths, zero orphans. The inventory is genuinely current.

## What the correctness work changed about the release claim

`a3dcbd79` matters for the truthfulness of generated documentation.
`docs/capabilities.md` is rendered from the capability statics and already
advertised `training = yes` for `prod_all`, `prod_dim`, and `cumsum`, and the
operation catalog already declared `GradientRule::Defined` for the Reduction
and Scan identities. The CPU kernels recorded no tape entry, so a backward pass
through any of the three stopped silently and the generated table was making a
promise the backend did not keep. It was closed in the backend rather than by
narrowing the advertised row, so the generated documentation is now truthful
for those rows without a documentation edit. `0031816a` is the same shape of
defect in training-mode batch norm.

## Public API freeze

`docs/public-api/API_TIERS.md` is new and classifies every module in `incin`,
`incin-core`, `incin-backends`, and `incin-viz` as stable user API,
expert/backend-authoring API, intentional macro ABI, or preview.

One deliberate removal: `incin_backends::iteration`, which exposed only
`tile_2d`, a 2D loop-tiling helper taking runtime dimensions and sitting below
the descriptor contract. `tools/check-public-api.sh` gained an assertion that
the module stays `pub(crate)`; that guard was negative-tested by reverting the
visibility and confirming the gate fails.

`simd` and `codegen` were reviewed as the same kind of candidate and
deliberately kept, since neither executes operations nor bypasses a capability
query. Recording that decision was the point of the exercise: an undocumented
surface frozen by accident is the failure mode, and the same surface frozen
deliberately is not.

Baseline effect: `incin-backends` moved from 1111 to 1109 entries. No other
baseline changed.

## Two defects the integration run found that neither line of work had alone

**Formatting.** Each line passed `cargo fmt --all -- --check` on its own and
the merged tree failed it. Fixed in `53a084d6`, verified semantics-preserving
by comparing the affected files with whitespace and trailing commas stripped.

**`tools/command-smoke.sh` could not pass on a fresh clone.** Its `inspect`
check required `mnist_model.safetensors` or `rnn_model.safetensors` at the
repository root, but `*.safetensors` is gitignored, so those files exist only
in a tree that has already run one of the examples. The check failed with no
defect present, which meant `tools/ci-local.sh` could not pass anywhere except
a machine that happened to have run the RNN example. Fixed in `69eeb434` by
preferring a real checkpoint when present and otherwise synthesizing a minimal
well-formed container in the script's scratch directory. The assertion is
unchanged and both paths were exercised.

No GitHub workflow runs `command-smoke.sh`; only `tools/ci-local.sh` does. So
this was never breaking CI, and it was also never being caught by CI.

## How the release identity chain was verified

`tools/release-preflight.py` was exercised against a throwaway local `v0.1.0`
tag, deleted immediately and never pushed. Both directions were checked:

- tag at HEAD: `release preflight passed`
- tag one commit behind HEAD: refused, naming both the checkout and the commit
  the tag resolved to

So the gate verifies the exact commit being released, not merely a well-formed
tag name. `.github/workflows/release.yml` checks out the tag ref, requires a
manual publish dispatch to come from the matching tag, and passes
`--base-ref origin/master` for ancestry. All four workflows use SHA-pinned
actions.

## Known deviations and actions required before tagging

**The candidate is not reachable from `master`, so the release gate will refuse
a tag as things stand.** `.github/workflows/release.yml` runs
`tools/release-preflight.py --base-ref origin/master`, and that calls
`git merge-base --is-ancestor <tag commit> origin/master`. The candidate lives
on `integration/0.1.0` while `master` is still at `094bb484`, so the check
fails with `release tag 'v0.1.0' is not reachable from master`. This was
confirmed by running the gate with the base-ref CI actually uses, not with a
base-ref that is trivially satisfied.

This is a property of where the work sits, not a defect in the work. Before a
tag can be cut, `integration/0.1.0` has to be merged into `master`, or the
release workflow has to be pointed at a different base ref. Merging to `master`
is a maintainer decision and was deliberately not done here.

`CHANGELOG.md` heads the section `## [0.1.0] - 2026-08-24` and the candidate
was assembled on 2026-08-25. Nothing gates on that date, so it was left rather
than guessed at a second time; it should be set to the day the tag is cut.

Four commit subjects in the correctness stream exceed the repository's
72-character convention (76, 74, 73, and 73 characters). They were left intact
rather than rewritten, because rewriting another author's commits would change
their identity and break the relationship to the branch they came from. This is
flagged for the maintainer rather than silently corrected.

The release tag does not exist yet. `v0.1.0-rc.1` points at `71b3485c`, an
ancestor of this candidate. `cargo-semver-checks` records a skip rather than a
pass until a release tag exists, which is correct for a first release.

Nothing in this repository has been pushed. All work is local.

## Evidence index

- [`environment.txt`](environment.txt)
- [`validation.txt`](validation.txt)
- [`known-limitations.md`](known-limitations.md)
