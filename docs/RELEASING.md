# Releasing Incin

Releases are created automatically by GitHub Actions when a tag beginning with
`v` is pushed. The workflow can also be started manually for an existing tag.

```bash
git tag v0.1.0
git push origin v0.1.0
```

Each release contains:

- the rendered mdBook as a tarball and a single-page HTML file;
- the VS Code `.vsix` extension;
- Neovim and RustRover integration archives;
- `incin-lsp` and `cargo-incin` binaries for Linux x86_64, macOS Apple Silicon,
  and Windows x86_64.

The RustRover archive contains the supported External Tool and File Watcher
integration. Native JetBrains LSP support is not included because it is not a
verified integration in this repository.

## Publishing to crates.io

The workspace is at `0.1.0` and every publishable crate carries the metadata
crates.io and docs.rs require; `publish-metadata` in CI is what keeps that
true. Publication is still a manual step, and it has to run in dependency
order, because each crate resolves its path dependencies against the registry
versions its manifest names:

```bash
for package in incin-macros incin-core incin-telemetry incin-viz-plugin-api \
               incin-backends incin-data incin-diagnostics incin-lsp \
               incin-viz incin; do
  cargo publish -p "$package" --locked
done
```

A crate cannot be published until everything it depends on is already on the
registry, so the first release has to go one at a time and wait for the index
between steps. `cargo publish --dry-run` is not a useful pre-flight before the
first release: it resolves path dependencies against registry versions that do
not exist yet, and fails for that reason rather than for anything about the
manifests. `tools/check-publish-metadata.py` is the check that does work
beforehand.

Building `incin-core` no longer requires `protoc`. The ONNX protobuf module is
checked in at `crates/incin-core/src/generated/onnx.rs`, regenerated with
`cargo xtask onnx`, and verified against `proto/onnx.proto` by `cargo xtask
onnx --check` in CI (the only job in the repository that installs a
protobuf compiler.
