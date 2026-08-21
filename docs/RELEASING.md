# Releasing Incin

Pushing a tag beginning with `v` builds, verifies, and uploads a **draft**
release. The tagged commit must already be reachable from `master`. The
workflow does not make the release public. Inspect the draft and its checksums
before using the explicit manual publish action for the same tag.

Release candidates use the same workflow and assets as the final release. Set
the workspace, internal path-dependency requirements, VS Code package, and npm
lockfile to the full candidate version before creating tags such as
`v0.1.0-rc.1`. The workflow marks those GitHub releases as prereleases. Restore
the package version to `0.1.0` in a reviewed commit before tagging the final
release; do not reuse a candidate tag or replace its assets.

```bash
git tag v0.1.0
git push origin v0.1.0
```

In the Actions UI, select the release tag itself as the workflow ref, enter the
same tag, and choose `publish-existing-draft`. This action downloads the
uploaded draft assets and verifies their manifest, checksums, and GitHub asset
list again. It then waits at the `release` environment, so any environment
protection configured in GitHub applies before the verification and
`gh release edit --draft=false` steps run. The repository's environment policy
accepts `v*` tag refs, not `master`, for this job.

The equivalent GitHub CLI command is:

```bash
gh workflow run release.yml --ref v0.1.0 \
  -f tag=v0.1.0 \
  -f action=publish-existing-draft
```

Do not select `build-draft` for a tag that already has a draft: the workflow
intentionally refuses to alter an existing release. Use it only to package a
tag whose draft does not exist yet.

Each release contains:

- the rendered mdBook as a tarball and a single-page HTML file;
- the VS Code `.vsix` extension;
- Neovim and RustRover integration archives;
- `incin-lsp` and `cargo-incin` binaries for Linux x86_64, macOS Apple Silicon,
  and Windows x86_64.

The RustRover archive contains the supported External Tool and File Watcher
integration. Native JetBrains LSP support is not included because it is not a
verified integration in this repository.

The workflow does not publish crates to crates.io, the VS Code extension to
the Visual Studio Marketplace or Open VSX, or the Neovim module to a plugin
registry. Those are separate distribution steps. Until a GitHub release is
published, its listed assets do not exist for users to download.

`incin-lsp` is a standalone crates.io package. `cargo-incin` is the
`cargo-incin` binary in the `incin` package, so users install it with
`cargo install incin --bin cargo-incin` after the registry publication. For a
checkout before that publication, use `cargo install --path crates/incin --bin
cargo-incin --locked`. The equivalent `incin-lsp` commands are `cargo install
incin-lsp` after publication and `cargo install --path crates/incin-lsp --bin
incin-lsp --locked` from a checkout.

## Publishing to crates.io

The workspace is at `0.1.0` and every publishable crate carries the metadata
crates.io and docs.rs require; `publish-metadata` in CI is what keeps that
true. Publication is still a manual step, and it has to run in dependency
order, because each crate resolves its path dependencies against the registry
versions its manifest names:

Use the release helper rather than a loop. It verifies the fixed ten-package
order, package metadata and file lists, internal release-version dependencies,
the workspace version against the tag, and the current crates.io state. It
writes a non-secret report containing public crates.io owner logins, the
configured docs.rs feature set, and the expected documentation artifact URL.
`check` never publishes:

```bash
python3 tools/publish-crates.py check --tag v0.1.0 --report release-publish-report.json
```

For publication, check out the clean tag at a commit reachable from
`origin/master`. Each command requires the exact next package as an explicit
confirmation, publishes only that package, and waits for both the crates.io API
and Cargo's resolver before returning. Publication explicitly targets the
`crates-io` registry. Re-run it with the next listed package; an already visible
prefix is skipped without invoking `cargo publish`.

```bash
python3 tools/publish-crates.py publish --tag v0.1.0 --confirm incin-macros \
  --report release-publish-report.json
```

A crate cannot be published until everything it depends on is already on the
registry, so the first release has to go one at a time and wait for the index
between steps. `cargo publish --dry-run` is not a useful pre-flight before the
first release: it resolves path dependencies against registry versions that do
not exist yet, and fails for that reason rather than for anything about the
manifests. `tools/check-publish-metadata.py` is the check that does work
beforehand.

After every package is visible, use the helper's isolated smoke gate. It uses a
fresh `CARGO_HOME`, installs the exact release versions from crates.io, verifies
that `incin-lsp` installs only its executable, installs `cargo-incin` from the
`incin` package and runs the real `cargo incin doctor` subcommand, then checks a
fresh `incin = "=0.1.0"` consumer. It also refuses verification until every
documentable package has a live docs.rs artifact and every package has a
non-empty public owner list:

```bash
python3 tools/publish-crates.py verify --tag v0.1.0 --smoke \
  --report release-publish-report.json
```

Do not mark the release complete until this registry-installed smoke test
works.

Building `incin-core` no longer requires `protoc`. The ONNX protobuf module is
checked in at `crates/incin-core/src/generated/onnx.rs`, regenerated with
`cargo xtask onnx`, and verified against `proto/onnx.proto` by `cargo xtask
onnx --check` in CI (the only job in the repository that installs a
protobuf compiler).
