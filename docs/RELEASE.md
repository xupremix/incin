# Release integrity

> Operator procedure lives in [`RELEASING.md`](./RELEASING.md); this file
> defines the gates the procedure operates. Read both before tagging.

`release.yml` is a gated artifact pipeline. A push still wakes the workflow for
`v*` tags, but its preflight accepts only `vMAJOR.MINOR.PATCH` tags, optionally
with a SemVer prerelease suffix. It checks that the runner is checked out at
the tag's peeled commit, that the commit is reachable from `master`, that every
publishable workspace Cargo package has the tag version, and that
`editors/vscode/package.json` has that same version. A prerelease tag such as
`v0.1.0-rc.1` therefore requires the workspace and editor package to be
`0.1.0-rc.1`; the final release commit changes them to `0.1.0` before the
stable tag is created.

All build jobs test and package into GitHub Actions workflow artifacts; they do
not contact a GitHub release. The book job runs both its static checker and the
real Chromium browser test. The VS Code job runs its existing real VS Code test
harness before `vsce` packages the committed package version. The workflow
never rewrites the npm version while building.

The verification job downloads the complete platform matrix, writes an exact
expected-assets manifest, generates SHA-256 checksums, and rejects missing,
extra, or altered assets. Only then does the tag workflow create a draft
release, upload the verified files, and query the draft's uploaded asset list.
It never publishes that draft. Publication is a separate manual
`workflow_dispatch` action for the same tag: it downloads the draft assets,
rebuilds the expected manifest, verifies the checksums and the draft asset
list, and runs in the `release` environment before making the draft public.
Tags with a SemVer prerelease suffix are marked as prereleases in GitHub.
The dispatch must use the release tag as its workflow ref, matching the
environment's `v*` tag policy.
The RustRover archive is deliberately named
`incin-rustrover-external-tool-<version>.tar.gz`: the supported integration is
the verified External Tool/File Watcher fallback, not native LSP integration.

Run the local structural guard after editing the release workflow:

```sh
python3 tools/check-release-workflow.py
```
