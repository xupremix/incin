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

The workspace Cargo version is still development-only (`0.0.0`), so these are
GitHub release artifacts rather than crates.io publication packages.
