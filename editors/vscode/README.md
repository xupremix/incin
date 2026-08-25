# Incin Shape Diagnostics (VS Code)

Humanizes Incin's compile-time shape errors and adds real tensor-shape inlay
hints, without forking or replacing rust-analyzer.

## What it does

This extension does not speak LSP itself and contains no typenum-parsing
logic; it only points the standard rust-analyzer extension's server binary
at `incin-lsp`, a thin proxy that spawns the real rust-analyzer and rewrites
diagnostics, inlay hints, and hover labels through the `incin-diagnostics`
crate before they reach you:

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
the locally installed `incin-lsp`, and rust-analyzer. The automated editor test
also checks the complete path for diagnostics, hints, and completions.

## Requirements

- `incin-lsp` on your `PATH` (`cargo install incin-lsp`, or
  `cargo install --path crates/incin-lsp --bin incin-lsp --locked` from a
  checkout), or set `incin.lspPath` to its absolute path. A normal
  `cargo install incin-lsp` installs only the proxy executable.
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

The extension also sets `rust-analyzer.inlayHints.maxLength` to `null` for the
workspace. rust-analyzer otherwise truncates the type before `incin-lsp` sees
it, leaving an incomplete `DimCons<UInt<…>>` label that cannot be translated
reliably. Incin-lsp rewrites the complete type back to a compact shape label.

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

This launches a real VS Code Extension Development Host via
`@vscode/test-electron`, which downloads its own VS Code build the first time.
The host loads this extension from source, opens a throwaway workspace
containing a `Cargo.toml` that mentions `incin`, and asserts the extension
activates and correctly rewrites `rust-analyzer.server.path`/`extraEnv`, and
that **Incin: Toggle Shape Hints** flips the hints env var. It installs
`rust-lang.rust-analyzer` into its own isolated test profile first, which this
extension's `extensionDependencies` in `package.json` requires. It does not
touch your real VS Code profile. The standard run keeps the fast
activation/settings tests only.

Do not point the harness at a snap-packaged `code`. That build was confirmed
to silently swallow `--extensionTestsPath` and exit 0 without running
anything.

The dedicated pipeline check used in CI builds `incin-lsp`, opens a local Incin
workspace, and waits (at most two minutes) for a humanized rust-analyzer
diagnostic and a full tensor inlay hint; it also asserts that a completion
request still succeeds through the proxy. It pins VS Code 1.134.0 and
rust-analyzer 0.3.2971:

```bash
cargo build -p incin-lsp
INCIN_REAL_E2E=1 \
INCIN_E2E_LSP_PATH="$PWD/../../target/debug/incin-lsp" \
INCIN_E2E_REPO_ROOT="$PWD/../.." \
xvfb-run -a npm test
```

For an offline or repeatable run, set `INCIN_TEST_RA_VSIX` to a downloaded
rust-analyzer VSIX. By default the harness installs pinned rust-analyzer
0.3.2971 into its isolated profile.

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
