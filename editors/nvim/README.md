# kindle-lsp for Neovim

Routes Neovim's `rust_analyzer` LSP client through the `kindle-lsp` proxy, so
shape errors and inlay hints show up humanized (decimal shapes instead of
`UInt<...>` walls) — see `docs/growth/02-ide-extensions.md` for the
architecture. This is a ~40-line Lua module with **no typenum-parsing logic
of its own**; all humanization happens in the `kindle-diagnostics` Rust crate
behind `kindle-lsp`.

Targets Neovim's native `vim.lsp.config`/`vim.lsp.enable` (0.11+). If you're
on nvim-lspconfig instead (older Neovim, or you just prefer it), see
"Manual / nvim-lspconfig install" below — `M.server_opts()` returns a plain
table that works with either.

## Requirements

- Neovim 0.11+ (for `vim.lsp.config`/`vim.lsp.enable`), **or**
  nvim-lspconfig on any supported Neovim version.
- `kindle-lsp` on your `$PATH` (`cargo install --path crates/kindle-lsp --bin
  kindle-lsp` from the Kindle repo — the explicit `--bin` matters: the crate
  also builds a `mock-rust-analyzer` test fixture that you don't want on your
  `PATH`), or pass `lsp_path` to `setup()`/`server_opts()`.

## lazy.nvim

```lua
{
  dir = "/path/to/kindle/editors/nvim", -- or your own fork's URL
  name = "kindle-lsp",
  ft = "rust",
  config = function()
    require("kindle-lsp").setup()
  end,
},
```

## Manual install

Copy `lua/kindle-lsp.lua` onto your `runtimepath` (e.g.
`~/.config/nvim/lua/kindle-lsp.lua`), then in `init.lua`:

```lua
require("kindle-lsp").setup()
```

## nvim-lspconfig (older Neovim)

```lua
require("lspconfig").rust_analyzer.setup(require("kindle-lsp").server_opts())
```

## Options

`setup()` and `server_opts()` both take the same optional table:

```lua
require("kindle-lsp").setup({
  lsp_path = "kindle-lsp",   -- default: resolved via $PATH
  hints_enabled = true,      -- default: true — set false to disable hint rewriting
  shorten_backend = false,   -- default: false — true drops the backend/dtype/grad tail
})
```

To toggle at runtime, call `setup()` again with new options and restart the
client (`:LspRestart`) — `kindle-lsp` reads its config from the environment
once at startup.
