# Incin Shape Diagnostics (VS Code)

Humanizes Incin's compile-time shape errors and adds real tensor-shape inlay
hints, without forking or replacing rust-analyzer.

## What it does

This extension does not speak LSP itself and contains no typenum-parsing
logic; it only points the standard rust-analyzer extension's server binary
at `incin-lsp`, a thin proxy that spawns the real rust-analyzer and rewrites
two message kinds through the `incin-diagnostics` crate before they reach
you:

**Before** (raw rustc/rust-analyzer output):
```
error[E0271]: type mismatch resolving `<(...) as ElementCount>::Count == UInt<UInt<UInt<UTerm, B1>, B1>, B0>`
```

**After** (with this extension active):
```
error: Cannot reshape: source has 6 elements but the target shape has 8 elements
```

Hovering an intermediate tensor shows `Tensor<[32, 128]>` instead of
`Tensor<(UInt<...>, UInt<...>), CpuBackendImpl<f32, Cpu>>`.

![A reshape error rewritten by incin-lsp in VS Code](../../docs/assets/editors/vscode-shape-diagnostic.png)

This capture comes from an isolated VS Code profile running the packaged VSIX,
the locally installed `incin-lsp`, and rust-analyzer. The extension's automated
test covers activation and settings. The live capture checks the complete
diagnostic path.

## Requirements

- `incin-lsp` on your `PATH` (`cargo install incin-lsp` after the first
  crates.io publication, or `cargo install --path crates/incin-lsp --bin
  incin-lsp --locked` from a checkout before then), or set `incin.lspPath` to
  its absolute path. A normal `cargo install incin-lsp` installs only the
  proxy executable.
- The standard `rust-lang.rust-analyzer` extension installed and enabled;
  this extension configures it, it does not replace it.

## How it works

On activation (only in workspaces whose `Cargo.toml` mentions `incin`), this
extension sets:
- `rust-analyzer.server.path` → your configured `incin.lspPath`
  (default `incin-lsp`, resolved via `PATH`).
- `rust-analyzer.server.extraEnv` → merges in `INCIN_LSP_HINTS` /
  `INCIN_LSP_SHORTEN_BACKEND` so the **Incin: Toggle Shape Hints** command
  can flip hint rewriting without you editing settings by hand. When the
  official extension includes a bundled server binary, it also sets
  `INCIN_LSP_RA_PATH` to that binary.

`server.extraEnv` is a best-effort integration point: it's used here because
it's the standard place rust-analyzer's own extension exposes environment
overrides for the server process, but if a future version of that extension
renames or removes it, `server.path` alone still gets you humanized
diagnostics and hints (incin-lsp's shipped defaults: hints on, backend tail
kept). Toggling would just stop working until updated to match.

## Settings

| Setting | Default | Description |
|---|---|---|
| `incin.lspPath` | `"incin-lsp"` | Path to the incin-lsp binary. |
| `incin.shortenBackend` | `false` | Drop the backend/dtype/grad tail from inlay hints. |

## Commands

- **Incin: Toggle Shape Hints**: flips inlay-hint rewriting and restarts
  rust-analyzer so the change takes effect.

## Running the tests

```bash
npm ci
npm test
```

This launches a real VS Code Extension Development Host (via
`@vscode/test-electron`, which downloads its own VS Code build the first
time. Do not point it at a snap-packaged `code`; that was confirmed to
silently swallow `--extensionTestsPath` and exit 0 without running anything)
with this extension loaded from source, opens a throwaway workspace
containing a `Cargo.toml` that mentions `incin`, and asserts the extension
activates and correctly rewrites `rust-analyzer.server.path`/`extraEnv`, and
that **Incin: Toggle Shape Hints** flips the hints env var. It installs
`rust-lang.rust-analyzer` into its own isolated test profile first (needed
because of `extensionDependencies` above). It does not touch your real VS
Code profile. It does **not** spin up a full `incin-lsp`/rust-analyzer
workspace. The live check shown above covers that boundary.

For an offline or repeatable run, set `INCIN_TEST_RA_VSIX` to a downloaded
rust-analyzer VSIX. By default the harness installs the current Marketplace
version into its isolated profile.

## Building from source

```bash
npm ci
npm run compile
npx @vscode/vsce package
```

This produces `incin-lsp-vscode-<version>.vsix` in this directory. Install it
into VS Code with either:

```bash
code --install-extension incin-lsp-vscode-0.1.0.vsix
```

or, from the UI: Extensions view → `...` menu → **Install from VSIX...** →
select the file. Reload the window afterwards. Uninstall the same way as any
other extension (Extensions view → Incin Shape Diagnostics → Uninstall).
