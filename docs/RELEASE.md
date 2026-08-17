# Release integrity

`release.yml` is a gated artifact pipeline. A push still wakes the workflow for
`v*` tags, but its preflight accepts only `vMAJOR.MINOR.PATCH` tags, optionally
with a SemVer prerelease suffix. It checks that the runner is checked out at
the tag's peeled commit, that every publishable workspace Cargo package has the
tag version, and that `editors/vscode/package.json` has that same version.

All build jobs test and package into GitHub Actions workflow artifacts; they do
not contact a GitHub release. The book job runs both its static checker and the
real Chromium browser test. The VS Code job runs its existing real VS Code test
harness before `vsce` packages the committed package version. The workflow
never rewrites the npm version while building.

The verification job downloads the complete platform matrix, writes an exact
expected-assets manifest, generates SHA-256 checksums, and rejects missing,
extra, or altered assets. Only then does the workflow create a draft release,
upload the verified files, query the draft's uploaded asset list, and publish
it after a second verification. The RustRover archive is deliberately named
`incin-rustrover-external-tool-<version>.tar.gz`: the supported integration is
the verified External Tool/File Watcher fallback, not native LSP integration.

Run the local structural guard after editing the release workflow:

```sh
python3 tools/check-release-workflow.py
```
