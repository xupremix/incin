-- Configures Neovim's `rust_analyzer` LSP client to launch through the
-- incin-lsp proxy instead of talking to rust-analyzer directly. Contains no
-- typenum-parsing logic itself; all humanization lives in the
-- incin-diagnostics Rust crate behind incin-lsp. See
-- docs/growth/02-ide-extensions.md for the full architecture.
--
-- Targets Neovim's native `vim.lsp.config`/`vim.lsp.enable` (0.11+): it
-- merges onto whichever base `rust_analyzer` config is already registered
-- (Neovim's bundled default, or nvim-lspconfig's, if installed) rather than
-- replacing it, so filetype/root-dir detection etc. keep working; only
-- `cmd`/`cmd_env` are overridden here.
--
-- Important, confirmed against a real nvim-lspconfig v2.11 install: if your
-- config calls `require("lspconfig").rust_analyzer.setup(...)` yourself
-- (directly, or indirectly via `mason-lspconfig.setup({ handlers = {...} })`
-- the common kickstart.nvim-derived pattern), that goes through
-- nvim-lspconfig's own legacy manager (`lspconfig/configs.lua`), which does
-- **not** call `vim.lsp.config`/`vim.lsp.enable` at all. `M.setup()` below is
-- a no-op in that case; use `M.merge_into()` instead, inside whatever
-- table *you* pass to that `.setup()` call.

local M = {}

--- @class IncinLspOpts
--- @field lsp_path? string path to the incin-lsp binary (default: "incin-lsp", resolved via $PATH)
--- @field hints_enabled? boolean whether inlay-hint labels get rewritten (default: true)
--- @field shorten_backend? boolean drop the backend/dtype/grad tail from rewritten hints (default: false)

--- Builds the `cmd`/`cmd_env` override table for launching rust-analyzer
--- through incin-lsp. Useful standalone if you configure `rust_analyzer`
--- yourself (e.g. via nvim-lspconfig's `.setup{}` or `vim.lsp.start`)
--- instead of calling `setup()` below.
--- @param opts? IncinLspOpts
--- @return table
function M.server_opts(opts)
  opts = opts or {}
  local hints_enabled = opts.hints_enabled
  if hints_enabled == nil then
    hints_enabled = true
  end

  return {
    cmd = { opts.lsp_path or "incin-lsp" },
    cmd_env = {
      INCIN_LSP_HINTS = hints_enabled and "1" or "0",
      INCIN_LSP_SHORTEN_BACKEND = (opts.shorten_backend and "1") or "0",
    },
    -- rust-analyzer truncates labels before incin-lsp receives them unless
    -- this is unlimited. A partial `DimCons<UInt<…, …>, …>` label cannot be
    -- reconstructed safely by the proxy.
    settings = {
      ["rust-analyzer"] = {
        inlayHints = { maxLength = vim.NIL },
      },
    },
  }
end

--- One-call setup: merges the incin-lsp override onto the `rust_analyzer`
--- config and enables it. Call this from your Neovim config instead of (not
--- in addition to) your own `rust_analyzer` setup call.
---
--- Only takes effect if `rust_analyzer` is actually started via
--- `vim.lsp.enable` (Neovim's native 0.11+ mechanism, or a plain
--- `vim.lsp.start` call). If your config drives `nvim-lspconfig`'s own
--- `require("lspconfig").rust_analyzer.setup(...)` directly or through
--- `mason-lspconfig`'s `handlers` bypasses this entirely; use
--- `M.merge_into()` instead.
--- @param opts? IncinLspOpts
function M.setup(opts)
  vim.lsp.config("rust_analyzer", M.server_opts(opts))
  vim.lsp.enable("rust_analyzer")
end

--- Merges the incin-lsp `cmd`/`cmd_env` override into an existing
--- nvim-lspconfig-shaped server config table for `mason-lspconfig`
--- `handlers`, a direct `require("lspconfig").rust_analyzer.setup(...)`
--- call, or anywhere else you already build that table yourself instead of
--- calling `setup()` above. Does not mutate `server`; returns a new table.
---
--- Example, inside a `mason-lspconfig.setup({ handlers = { ... } })` handler:
--- ```lua
--- handlers = {
---   function(server_name)
---     local server = servers[server_name] or {}
---     if server_name == "rust_analyzer" then
---       server = require("incin-lsp").merge_into(server)
---     end
---     require("lspconfig")[server_name].setup(server)
---   end,
--- }
--- ```
--- @param server? table an existing nvim-lspconfig-shaped server config
--- @param opts? IncinLspOpts
--- @return table
function M.merge_into(server, opts)
  return vim.tbl_deep_extend("force", server or {}, M.server_opts(opts))
end

return M
