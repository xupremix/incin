# Agent Skills & IDE Setup

Incin provides first-class developer tooling for both human developers and autonomous AI coding agents.

---

## 1. Incin AI Agent Skills

Incin packages specialized skills for LLM-powered coding assistants:

| Skill Name | Purpose | Target Audience |
| :--- | :--- | :--- |
| **`incin-expert`** | Building models, neural networks, training pipelines, and shape algebra with Incin | Library Users & ML Engineers |
| **`incin-engineering`** | Implementing internals, custom kernels, backends, and frozen contracts | Framework Core Developers |
| **`incin-repository`** | Navigating repository, building docs, running CI, and testing | Repo Contributors |

### Installing Skills into your Agent Environment

You can install Incin skills into your project with a single command using `cargo-incin`:

```bash
# Install to your current workspace
cargo incin skills install

# Or install specifically for your assistant tool
cargo incin skills install --tool cursor      # Cursor (.cursor/rules/)
cargo incin skills install --tool antigravity # Google Antigravity (.agents/skills/)
cargo incin skills install --tool claude     # Claude Code (.claude/skills/)
cargo incin skills install --tool windsurf   # Windsurf (.windsurf/rules/)
```

Alternatively, install using the universal shell script:

```bash
curl -fsSL https://raw.githubusercontent.com/xupremix/incin/master/tools/install-skills.sh | bash
```

---

## 2. Editor Integrations (LSP Proxy)

Incin translates complex `typenum` type errors into clean, intuitive shape diagnostics in your editor.

### VS Code
Install the official extension from the marketplace or local package:
1. Extension: `incin-lsp-vscode`
2. Routes `rust-analyzer` through `incin-lsp` proxy binary.
3. Automatically shortens verbose type-level inlay hints into clean `Tensor<[2, 3]>` labels.

### Neovim
Add the following to your `init.lua`:

```lua
require("incin-lsp").setup({
    lsp_path = "incin-lsp", -- resolved via PATH
    shorten_backend = true,
})
```

### RustRover / JetBrains
Configure External Linters -> Cargo to use `cargo-incin check` instead of standard `cargo check`.
