# incin-lsp

`incin-lsp` is a small proxy for rust-analyzer. It keeps rust-analyzer as the
language server and rewrites Incin's type-level tensor diagnostics and inlay
hints into readable shapes before they reach your editor.

It only changes these server-to-editor messages:

- pushed and pulled diagnostics, where typenum expressions become decimal
  shapes and known failures gain focused guidance;
- responses to `textDocument/inlayHint`, where tensor labels are simplified;
- responses to `textDocument/hover`, where displayed type signatures are
  simplified.

All other LSP frames are forwarded unchanged. The proxy does not implement a
separate language server and does not replace rust-analyzer.

## What it looks like

A reshape error leaves rust-analyzer in this form:

```text
Cannot reshape: source has UInt<UInt<UInt<UTerm, B1>, B1>, B0> elements but
the target shape has UInt<UInt<UInt<UInt<UTerm, B1>, B0>, B0>, B0> elements
```

and reaches your editor in this one, with the original expansions attached as
related information rather than thrown away:

```text
Cannot reshape: source has 6 elements but the target shape has 8 elements

  6 <= UInt<UInt<UInt<UTerm, B1>, B1>, B0>
  8 <= UInt<UInt<UInt<UInt<UTerm, B1>, B0>, B0>, B0>
```

Inlay hints and hover labels get the same treatment, so

```text
Tensor<(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>), CpuBackendImpl<Cpu>>
```

reads as `Tensor<[2, 3], CpuBackendImpl<Cpu>>`, or as `Tensor<[2, 3]>` with
`INCIN_LSP_SHORTEN_BACKEND=1`.

## Install

After the first crates.io release, install the binary with:

```bash
cargo install incin-lsp
```

Until then, install it from a checkout:

```bash
cargo install --path crates/incin-lsp --bin incin-lsp --locked
```

Both forms install one executable, `incin-lsp`. It starts `rust-analyzer` from
your `PATH`; install rust-analyzer separately through rustup or your editor.

For a published tagged Incin release, the [GitHub release assets](https://github.com/xupremix/incin/releases)
also include prebuilt `incin-lsp` binaries for supported platforms. Use the
asset matching your operating system and architecture, then place it on your
`PATH`.

## Configuration

Configuration is read once when the proxy starts.

| Variable | Default | Effect |
| --- | --- | --- |
| `INCIN_LSP_RA_PATH` | `rust-analyzer` | Path or command name of the rust-analyzer executable to spawn. |
| `INCIN_LSP_HINTS` | `1` | Set to `0` to leave inlay hints and hover labels unchanged. Diagnostics are still rewritten. |
| `INCIN_LSP_SHORTEN_BACKEND` | `0` | Set to `1` to remove the backend, dtype, and gradient tail from rewritten tensor labels. |

For example:

```bash
INCIN_LSP_RA_PATH=/path/to/rust-analyzer incin-lsp
```

## Editor setup

The repository includes configuration for [VS Code](https://github.com/xupremix/incin/tree/master/editors/vscode)
and [Neovim](https://github.com/xupremix/incin/tree/master/editors/nvim).
Both integrations launch `incin-lsp` in place of rust-analyzer and preserve
your existing rust-analyzer settings.

The proxy itself has deterministic framing and rewrite tests. The VS Code
extension has an automated activation/configuration test. The repository also
records live diagnostic checks in VS Code and Neovim as release evidence.

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/xupremix/incin/blob/master/LICENSE_APACHE)
or [MIT license](https://github.com/xupremix/incin/blob/master/LICENSE_MIT) at
your option.
