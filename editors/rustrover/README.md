# Incin diagnostics for RustRover / IntelliJ

RustRover uses its own Rust engine, not rust-analyzer, so the `incin-lsp`
proxy (which wraps rust-analyzer specifically) doesn't transparently apply
here the way it does for VS Code/Neovim. Per
`docs/growth/02-ide-extensions.md` §02.7, there are two honest options:
**this directory ships the verified one**; the other is documented but
explicitly marked unverified so it isn't oversold.

## Requirements

- `cargo-incin` on your `PATH` (`cargo install incin --bin cargo-incin` after
  the first crates.io publication, or `cargo install --path crates/incin --bin
  cargo-incin --locked` from a checkout before then). `incin-check.sh` checks
  for this itself and prints an install command if it is missing, so this is
  not a silent failure; installing it first saves the round trip.
- A RustRover (or any IntelliJ-platform Rust IDE) install with **External
  Tools** and, optionally, **File Watchers** available; both are core
  platform features, no plugin required for Option A.

## Option A: External Tool + File Watcher (✅ verified, shipped here)

Runs `cargo incin check` and surfaces its already-humanized output in
RustRover's own tool window. Less magical than inline squiggles (no inlay
hints, no red underline at the exact token; just readable text in a panel),
but it needs no plugin, works on any RustRover version, and the script itself
(`incin-check.sh`) has been run against this repo as part of building this
integration.

**Setup** (`Settings/Preferences → Tools → External Tools → +`):

| Field | Value |
|---|---|
| Name | `Incin Check` |
| Program | `$ProjectFileDir$/editors/rustrover/incin-check.sh` |
| Arguments | `-p $ProjectFileDir$` *(adjust to the package you want checked)* |
| Working directory | `$ProjectFileDir$` |
| Output filters | (optional) add a filter matching `` `$FILE_PATH$:$LINE$:$COLUMN$` `` if you want output lines to become clickable file links |

Run it via **Tools → External Tools → Incin Check**, or bind it to a
keyboard shortcut (`Settings → Keymap → External Tools`). For check-on-save,
wrap the same script in a **File Watcher** (`Settings → Tools → File
Watchers → +custom`) triggered on `*.rs` file changes.

## Option B: Native LSP integration (⚠️ unverified; do not rely on this yet)

Newer JetBrains IDE platform versions expose an external-LSP-server API
(`com.intellij.platform.lsp`) that could, in principle, run `incin-lsp`
directly the same way VS Code/Neovim do, giving inline diagnostics and inlay
hints instead of a separate tool window.

**This has not been built or tested against a real RustRover install in this
repo**; doing so needs the IntelliJ Platform SDK, Gradle, and a specific
RustRover version to target, none of which were available while building
this integration. Before attempting it:
1. Confirm your installed RustRover version actually exposes the LSP API
   (it's gated by platform version, not universal).
2. Confirm RustRover's bundled Rust plugin doesn't already claim `.rs` files
   in a way that conflicts with a second LSP client attaching to them.

Until someone verifies this against a real install, **Option A is the
supported path**; do not advertise LSP-mode RustRover support in marketing
material ahead of that verification (see the DO-NOT list in
`docs/growth/02-ide-extensions.md`).
