# 02 — IDE Extensions + LSP (VS Code, Neovim, Rust Rover)

> **Depends on:** task `00` (`kindle-diagnostics`). Can run in parallel with `01`.
> **Effort:** Medium-High (the LSP proxy is the work; per-editor glue is thin).
> **Priority:** #2 — this is what makes Kindle *feel* magical and is the second
> flagship demo.

## Goal

Two capabilities, in every major editor, with zero per-editor duplication of
logic:

1. **Humanized diagnostics** — the `UInt<UInt<…>>` walls become `[32, 128]`
   *inline in the editor*, live, as you type (not just under `cargo kindle`).
2. **Shape inlay hints** — every intermediate tensor shows its real shape
   (`Tensor<[32, 128]>`) as a ghost-text hint, because the shape is a type the
   compiler already knows. This deletes the `print(x.shape)` workflow.

## The architectural key: one LSP proxy, thin editor clients

rust-analyzer already computes both diagnostics and type inlay hints — it just
renders typenum as `UInt<…>`. We do **not** fork rust-analyzer. Instead we ship
a tiny **LSP middleware proxy** (`kindle-lsp`) that sits between the editor and
rust-analyzer and rewrites the two relevant message types through
`kindle-diagnostics`. Every editor speaks LSP, so **one proxy serves all
editors**; each editor only needs ~30 lines of "launch this proxy instead of
`rust-analyzer` directly" config.

```
        ┌────────────┐   LSP over stdio   ┌───────────────┐   spawns   ┌──────────────┐
Editor ─┤ thin client├───────────────────►│  kindle-lsp   ├───────────►│ rust-analyzer│
(VSCode/│  (VSIX /   │◄───────────────────┤  (proxy)      │◄───────────┤ (child proc) │
 Neovim/│  Lua / XML)│  rewritten msgs    │  uses         │  raw msgs  └──────────────┘
 RustR.)└────────────┘                    │  kindle-      │
                                          │  diagnostics  │
                                          └───────────────┘
```

The proxy is transparent: it forwards every LSP message verbatim **except**:
- `textDocument/publishDiagnostics` — rewrite each diagnostic `.message` (and
  `relatedInformation`) through `kindle_diagnostics::humanize_diagnostic`.
- `textDocument/inlayHint` responses — rewrite each hint `.label` that contains
  a typenum shape into `[d0, d1, …]`.
- Optionally `textDocument/hover` — same rewrite on hover text.

Everything else (completion, goto-def, formatting) passes through untouched.

## Why a proxy and not a rust-analyzer plugin
rust-analyzer has no stable third-party plugin API for rewriting hint/diagnostic
rendering. A stdio LSP proxy is the stable, editor-agnostic seam, and it reuses
the exact same `kindle-diagnostics` formatter as the CLI (doc 00) — so the CLI,
the terminal, and every editor are guaranteed identical. This is the single
biggest reason task `00` exists.

## Component 1 — `kindle-lsp` (the proxy, Rust)

### Task 02.1 — scaffold the crate
- New crate `crates/kindle-lsp` (binary). Add to workspace members.
- Deps: `kindle-diagnostics` (path), a JSON-RPC/LSP transport. Use `tower-lsp`
  **only if** you implement server methods; for a pure pass-through proxy the
  lighter path is to read/write raw LSP frames (`Content-Length` framed JSON on
  stdio) and hand-parse the three message kinds we rewrite. Prefer the raw-frame
  approach: it means we never have to model the *entire* LSP surface, only
  forward bytes and intercept three method names. Document the choice in the
  crate's `lib.rs` header.

### Task 02.2 — the pass-through pump
- Spawn `rust-analyzer` as a child (respect `$RUST_ANALYZER_PATH`, else `PATH`).
- Pump A: editor stdin → child stdin, forwarding frames unchanged.
- Pump B: child stdout → editor stdout, **inspecting** each frame:
  - parse the `Content-Length` header, read the JSON body;
  - if `method == "textDocument/publishDiagnostics"`: for each diagnostic,
    replace `message` with `humanize_diagnostic(message).text`; if hint pairs
    exist, append them to `relatedInformation` so hovering shows the mapping.
  - if it is an `inlayHint` **response** (matched by request `id`, since
    responses have no `method`): rewrite each `label` (see Component 2).
  - re-serialize, recompute `Content-Length`, write out.
- **Critical correctness note:** `Content-Length` is in **bytes**, not chars.
  After rewriting, recompute from the serialized byte length or the client
  desyncs. Add a test that feeds a multi-byte-UTF-8 diagnostic and asserts the
  emitted header equals the new byte length.

### Task 02.3 — inlay-hint shape rewriting
- rust-analyzer type hints for a tensor render like `: Tensor<(UInt<…>, UInt<…>),
  CpuBackendImpl<f32, Cpu>>`. Add to `kindle-diagnostics` a focused helper:
  ```rust
  /// Rewrites a rust-analyzer inlay-hint label for a Kindle tensor into a
  /// compact shape form: `Tensor<(U2, U3), …>` → `Tensor<[2, 3]>`. Returns the
  /// input unchanged if it is not a Kindle tensor label.
  pub fn humanize_inlay_label(label: &str) -> String
  ```
  It (a) detects the `Tensor<( … ), …>` shell, (b) runs the tuple's typenum
  elements through the existing decimal translator, (c) drops the backend type
  params for brevity, producing `Tensor<[2, 3]>`. Unit-test it in
  `kindle-diagnostics` with real rust-analyzer label strings (capture a few from
  a live session and paste them as fixtures).

### Task 02.4 — config surface
- Env/flags: `KINDLE_LSP_RA_PATH` (rust-analyzer location),
  `KINDLE_LSP_SHORTEN_BACKEND=1` (drop `<f32, Cpu>` tails), `KINDLE_LSP_HINTS=0`
  to disable hint rewriting but keep diagnostics. Document in the crate header.

**Acceptance (proxy):** with `kindle-lsp` as the server, opening a file with a
shape error shows the humanized message; hovering an intermediate tensor shows
`Tensor<[…]>`; all other IDE features (completion, goto) still work. Add an
integration test that pipes a recorded rust-analyzer session through the proxy
and asserts the three message kinds are rewritten and everything else is
byte-identical.

## Component 2 — editor clients (thin)

Each client does one thing: **launch `kindle-lsp` as the Rust language server.**

### Task 02.5 — VS Code extension (`editors/vscode/`)
- Minimal `package.json` + `extension.ts`: on activate, if the workspace uses
  Kindle, set `rust-analyzer.server.path` to the bundled `kindle-lsp`, or
  register a language client that launches it. Ship `kindle-lsp` binaries per
  platform or require `cargo install kindle-lsp`.
- Add a command `Kindle: Toggle Shape Hints` flipping `KINDLE_LSP_HINTS`.
- Package with `vsce`. This is the flagship client — polish its README with the
  before/after screenshot from doc 01's demo.

### Task 02.6 — Neovim (`editors/nvim/`)
- Ship a Lua snippet / tiny plugin: configure `nvim-lspconfig` (or the built-in
  `vim.lsp.start`) to use `kindle-lsp` as the `rust_analyzer` `cmd`. Document
  both `lazy.nvim` and manual install. ~30 lines.

### Task 02.7 — Rust Rover / IntelliJ (`editors/rustrover/`)
- RustRover uses its own engine, not rust-analyzer, so the proxy trick does not
  transparently apply. **Two honest options — pick one and document the choice:**
  1. **LSP mode:** newer IntelliJ platforms support external LSP servers via the
     LSP API (`com.intellij.platform.lsp`). Ship a small plugin that registers
     `kindle-lsp`. Verify the target IntelliJ version supports it before
     committing to this path.
  2. **Fallback:** ship an *external tool* + file watcher that runs `cargo kindle
     check` and surfaces humanized output in a tool window. Less magical, but
     works on any IDE version.
  Do not overpromise RustRover parity in marketing until whichever path is
  actually verified on a real RustRover install.

## Verification
- `cargo test -p kindle-lsp` (proxy framing + rewrite tests).
- `cargo test -p kindle-diagnostics` (label/diagnostic humanization fixtures).
- Manual smoke per editor: open `crates/kindle-core/tests/compile_fail/
  reshape_static_mismatch.rs` in each editor with the client active; confirm the
  humanized message and a shape inlay hint appear.
- **Non-Rust builds:** VS Code — `npm ci && npm run compile && vsce package` in
  `editors/vscode/`. Document Node version.

## Risks / DO-NOT
- **DO-NOT** reimplement the typenum parser in TypeScript/Lua. All humanization
  is in `kindle-diagnostics` behind the proxy — the editor clients contain **no**
  parsing logic. This is the whole point of the architecture.
- **DO-NOT** break LSP framing: byte-accurate `Content-Length` after every
  rewrite (test with multi-byte UTF-8).
- **DO-NOT** claim RustRover support until path 02.7 is verified on real
  hardware; mark it "experimental" otherwise.
- **DO-NOT** bundle a GPU backend into `kindle-lsp` — it must start instantly.

## Demo script
Type a wrong reshape; the red underline says `Cannot reshape 6 → 8` instantly.
Then cursor down a forward pass and let the ghost-text shapes cascade —
`[32,784] → [32,128] → [32,10]` — with nothing running. Caption: *"I haven't run
anything. The editor already knows every shape."*

> **2026-07-23 status update:**
>
> **`kindle-lsp` (02.1–02.4): done, verified without a real editor attached.**
> Built as designed — raw `Content-Length` framing (`frame.rs`), pure JSON
> rewriting (`rewrite.rs`), env-based `Config` (`config.rs`), no
> `tower-lsp`/`lsp-types` dependency. One real bug caught by testing, not
> review: `humanize_inlay_label` (in `kindle-diagnostics`) left a dangling
> comma from Rust's 1-tuple syntax (`(U8,)` → `[8,]` instead of `[8]`) —
> fixed and pinned with a regression test. Verified three ways: unit tests on
> `frame`/`rewrite`/`config` in isolation; a `mock-rust-analyzer` test-only
> binary (`src/bin/mock_rust_analyzer.rs`, `test = false, doc = false` in
> `Cargo.toml` so it never ships) that stands in for a real rust-analyzer
> session; and one true end-to-end integration test
> (`tests/proxy_integration.rs`) that spawns the *actual* `kindle-lsp` binary
> against that mock server and asserts all three message kinds — an
> unrelated notification (byte-identical passthrough), a
> `publishDiagnostics` message (humanized + hints attached), and an
> `inlayHint` response correlated by request id (label shape-humanized).
> This satisfies the "pipe a recorded session through the proxy" acceptance
> criterion above without needing a real rust-analyzer installed anywhere,
> including in CI.
>
> **VS Code client (02.5): done, compiles clean, not run in a real VS Code.**
> `editors/vscode/` — real TypeScript, `npm install && npm run compile`
> succeeds under strict mode. Sets `rust-analyzer.server.path` (high
> confidence this setting exists and is stable) and
> `rust-analyzer.server.extraEnv` (used for the hints toggle — flagged in
> both the code and its README as a best-effort integration point to verify
> against whatever rust-analyzer extension version is actually installed,
> since it could not be confirmed against a live VS Code here). No before/after
> screenshot yet — needs a real editor session to capture honestly.
>
> **Neovim client (02.6): done, verified against a real Neovim 0.12.4.**
> `editors/nvim/lua/kindle-lsp.lua` targets the native `vim.lsp.config`/
> `vim.lsp.enable` (0.11+) rather than requiring nvim-lspconfig — checked
> both APIs' exact calling convention against the real, installed Neovim
> before writing this (its `vim.lsp.config` is a callable table via
> `__call`, not a plain function — easy to get wrong from memory alone) and
> then loaded the actual module headlessly (`nvim --headless -u NONE -c
> "lua require('kindle-lsp')..."`), asserting default/override option
> handling and that `setup()` runs without error. `server_opts()` still
> returns a plain table for nvim-lspconfig users on older Neovim.
>
> **RustRover (02.7): fallback only, shipped and tested; LSP mode still
> unverified — do not claim it works.** `editors/rustrover/kindle-check.sh`
> (wraps `cargo kindle check`) was actually executed against this repo.
> Building or testing a real IntelliJ-platform LSP plugin needs the
> IntelliJ Platform SDK, Gradle, and a specific RustRover version to target
> a real install against, none of which were available here — so, per this
> doc's own DO-NOT list, that path is documented as an option but explicitly
> marked unverified rather than shipped or claimed as working.
>
> **2026-07-23 follow-up: install-path review, two real bugs found and
> fixed.** Auditing all three clients end-to-end (not just re-reading the
> code) surfaced two install-time bugs neither the original build pass nor
> the test suite would have caught, since both are about what happens
> *outside* the process kindle-lsp/the extension itself runs in:
> 1. **`editors/vscode/package.json` was missing `extensionDependencies:
>    ["rust-lang.rust-analyzer"]`.** Without it, a user who installs the
>    Kindle extension without already having rust-analyzer installed gets no
>    prompt to install it — the extension silently writes
>    `rust-analyzer.server.path`/`server.extraEnv` into a settings namespace
>    nothing is listening to, with no error surfaced anywhere. Fixed.
> 2. **The documented install command, `cargo install --path
>    crates/kindle-lsp`, also installs `mock-rust-analyzer`** onto the
>    user's `PATH` — `cargo install` installs every `[[bin]]` target in a
>    package by default, and the crate's second bin (the test-only
>    rust-analyzer stand-in used by `tests/proxy_integration.rs`, gated
>    `test = false, doc = false` but *not* excluded from installation) is
>    exactly such a target. Considered gating it behind a non-default Cargo
>    feature instead, but that would require the workspace verification
>    loop (`docs/growth/README.md` §2) to special-case `kindle-lsp`'s
>    features to keep `cargo test` building the fixture at all — for a
>    one-binary packaging nicety, the lower-risk fix is the one shipped: the
>    VS Code and Neovim READMEs now both say `cargo install --path
>    crates/kindle-lsp --bin kindle-lsp` explicitly. Anyone later shipping
>    prebuilt `kindle-lsp` binaries (task 02.5's "per platform" option)
>    should exclude `mock-rust-analyzer` from that packaging the same way.
>
> Also completed as part of this pass, none of them bugs, just gaps: the VS
> Code README's "Building from source" section ended at `vsce package`
> without saying what to do with the resulting `.vsix` (added the
> `code --install-extension` step and the UI equivalent); the RustRover
> README had no "Requirements" section unlike its VS Code/Neovim siblings
> (added one, pointing at the CLI install below); and the root `README.md`
> gained `CLI` / `Editor / IDE Support` / `Documentation` sections plus
> `kindle-diagnostics`/`kindle-lsp` entries in the crate list — none of
> which had ever been added when those crates were built, so a newcomer
> reading only the root README had no way to discover any of this existed.
>
> **2026-07-23, same day, second follow-up: the VS Code client is now
> genuinely verified in a real VS Code — not just "compiles clean."** Both
> `code` and `rustrover` turned out to already be installed on this
> machine, which made real (not just compiled) verification possible for
> the first time. This was done with the user's explicit go-ahead after
> flagging upfront that testing this way would touch their actual VS Code
> profile (see the conversation for that checkpoint) — the end state
> leaves their profile exactly as it was: the test extension was
> uninstalled again afterward and confirmed absent via `code
> --list-extensions`.
>
> **What was added:** `editors/vscode/src/test/runTest.ts` +
> `src/test/suite/{index,extension.test}.ts`, using `@vscode/test-electron`
> to launch a real VS Code Extension Development Host with this extension
> loaded from source, open a throwaway temp workspace containing a
> `Cargo.toml` that mentions `kindle`, and assert two things through the
> real `vscode` API: (1) the extension activates and correctly rewrites
> `rust-analyzer.server.path`/`server.extraEnv`, and (2) the **`Kindle:
> Toggle Shape Hints`** command flips the hints env var. Run via `npm test`
> in `editors/vscode/`; `.vscode-test/` (the downloaded VS Code build this
> creates) is now gitignored.
>
> **Two real, non-obvious things this surfaced, worth knowing before anyone
> touches this again:**
> 1. **The system's snap-packaged `/snap/bin/code` silently swallows
>    `--extensionTestsPath`.** First attempt pointed `vscodeExecutablePath`
>    at it (to avoid a ~340 MB download) — the process exited 0 every
>    time, including with a *deliberately* broken assertion that must fail,
>    which is what caught this rather than a false "it works" being taken
>    at face value (file-based debug logging confirmed `run()` was never
>    even entered). Snap's confinement is the strongly-suspected cause.
>    **Fix:** let `@vscode/test-electron`'s `downloadAndUnzipVSCode()`
>    fetch its own unconfined build — this is also the tool's documented,
>    standard usage pattern, so fighting it to reuse the system snap
>    install wasn't the right call to begin with. `KINDLE_TEST_VSCODE_PATH`
>    still exists as an override for anyone with a non-snap `code` who
>    wants to skip the download.
> 2. **This accidentally became the real-world proof that the
>    `extensionDependencies` fix from the first 2026-07-23 follow-up
>    actually works.** The fresh test profile `test-electron` manages has
>    no extensions at all; the very first run failed with `Cannot activate
>    the 'Kindle Shape Diagnostics' extension because it depends on
>    unknown extension 'rust-lang.rust-analyzer'` — exactly the enforcement
>    that fix was meant to add, now confirmed by VS Code itself rather than
>    by reading the manifest schema. Fixed the test (not the extension) by
>    installing `rust-lang.rust-analyzer` into the test profile first via
>    `resolveCliArgsFromVSCodeExecutablePath` + `--install-extension`
>    before `runTests()`.
>
> Also added: `.vscodeignore` (there wasn't one; `vsce`/`@vscode/vsce` both
> warned about its absence, and it's now required anyway to keep
> `src/test/**` out of the shipped package); the README's build instructions
> now say `npx @vscode/vsce package`, not the deprecated unscoped `vsce`
> (confirmed the unscoped one is what `npx vsce` actually resolves to, with
> a `vsce has been renamed` deprecation warning to prove it).
>
> **Honest scope of what's now verified vs. still not:** this proves the
> extension's *own* logic — activation gating, config rewriting, the
> toggle command — works in a real VS Code. It does **not** exercise
> `kindle-lsp` itself or a real rust-analyzer diagnostic round-trip (the
> toggle test points `kindle.lspPath` at `/bin/true` specifically to avoid
> needing a real `kindle-lsp` binary on `PATH`, since that's out of scope
> for what this test is about). A full pipeline test — real `kindle-lsp`,
> real rust-analyzer indexing `kindle-core`, asserting on an actual
> humanized `publishDiagnostics` notification — is a meaningfully bigger,
> slower undertaking and remains future work, not something to silently
> claim was also covered here.
>
> RustRover: no new verification this pass. `rustrover` is also installed
> here, but Option B (native LSP plugin) still needs a Gradle/IntelliJ
> Platform SDK project that doesn't exist yet — installing the IDE itself
> doesn't change that. Nothing to update; Option A remains the only
> verified path.
