-- Configures Neovim's `rust_analyzer` LSP client to launch through the
-- kindle-lsp proxy instead of talking to rust-analyzer directly. Contains no
-- typenum-parsing logic itself — all humanization lives in the
-- kindle-diagnostics Rust crate behind kindle-lsp. See
-- docs/growth/02-ide-extensions.md for the full architecture.
--
-- Targets Neovim's native `vim.lsp.config`/`vim.lsp.enable` (0.11+): it
-- merges onto whichever base `rust_analyzer` config is already registered
-- (Neovim's bundled default, or nvim-lspconfig's, if installed) rather than
-- replacing it, so filetype/root-dir detection etc. keep working — only
-- `cmd`/`cmd_env` are overridden here.

local M = {}

--- @class KindleLspOpts
--- @field lsp_path? string path to the kindle-lsp binary (default: "kindle-lsp", resolved via $PATH)
--- @field hints_enabled? boolean whether inlay-hint labels get rewritten (default: true)
--- @field shorten_backend? boolean drop the backend/dtype/grad tail from rewritten hints (default: false)

--- Builds the `cmd`/`cmd_env` override table for launching rust-analyzer
--- through kindle-lsp. Useful standalone if you configure `rust_analyzer`
--- yourself (e.g. via nvim-lspconfig's `.setup{}` or `vim.lsp.start`)
--- instead of calling `setup()` below.
--- @param opts? KindleLspOpts
--- @return table
function M.server_opts(opts)
  opts = opts or {}
  local hints_enabled = opts.hints_enabled
  if hints_enabled == nil then
    hints_enabled = true
  end

  return {
    cmd = { opts.lsp_path or "kindle-lsp" },
    cmd_env = {
      KINDLE_LSP_HINTS = hints_enabled and "1" or "0",
      KINDLE_LSP_SHORTEN_BACKEND = (opts.shorten_backend and "1") or "0",
    },
  }
end

--- One-call setup: merges the kindle-lsp override onto the `rust_analyzer`
--- config and enables it. Call this from your Neovim config instead of (not
--- in addition to) your own `rust_analyzer` setup call.
--- @param opts? KindleLspOpts
function M.setup(opts)
  vim.lsp.config("rust_analyzer", M.server_opts(opts))
  vim.lsp.enable("rust_analyzer")
end

return M
