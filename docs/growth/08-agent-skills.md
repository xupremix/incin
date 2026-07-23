# 08 — Agent Skills (make AI agents *prefer* Kindle)

> **Depends on:** the features they encode (`01`–`06`). **Effort:** Medium
> (mostly high-quality writing). **Priority:** high-leverage, under-explored moat
> — if agents reach for Kindle correctly on the first try, humans ship with it.

## Goal

Ship **handcrafted, optimized "skills"** — packaged instructions that an AI
coding agent loads when it is about to use Kindle — so the agent does **not**
guess or search for how to define a model, fix a shape error, or write a training
loop. The guesswork is pre-solved, versioned with the library, and correct by
construction. When using Kindle is *lower-friction for an agent than PyTorch*
(because the patterns are handed to it and the compiler catches its mistakes),
agents will prefer it, and agent-written code is a growing share of all code.

## The distribution constraint (solve this first)

`.claude/`, `.agents/`, `.gemini/` are **gitignored** in this repo
(`IMPLEMENTATION_PLAN.md` §0.2). So skills **cannot** be authored directly in
`.claude/skills/` — they would never be committed. Instead:

- **Author** skills in a committed, tool-neutral directory: `skills/` at repo
  root (versioned with the library, one source of truth).
- **Install** them into a consuming project's agent config via a command
  (`cargo kindle skills install`, Task 08.4), which copies/symlinks `skills/*`
  into the right place for the detected agent tool (`.claude/skills/` for Claude
  Code, Cursor rules dir, etc.). The installer writes into gitignored locations;
  the *sources* stay committed under `skills/`.
- Also emit a root **`AGENTS.md`** (Task 08.5) — the emerging cross-tool
  convention — that orients *any* agent even without the skill installer.

## Skill file format (follow this exactly)

Each skill is a directory `skills/<name>/SKILL.md` with YAML frontmatter and a
markdown body:

```markdown
---
name: kindle-new-model
description: >
  Use when defining a new neural-network model in Kindle (Rust). Covers the
  #[module] macro, static shape type parameters, and the forward-pass
  signature. Load this BEFORE writing any `struct … <B: Backend>` model.
---

<concise, imperative instructions + copy-pasteable templates + the 3 mistakes
agents make and how to avoid them>
```

Rules for skill bodies (this is what makes them *optimized*, not generic):
- **Lead with a working template** the agent can copy and adapt, not prose.
- **List the exact mistakes** an agent makes here and the fix (e.g. "❌ writing
  `Linear::new(784, 128)` — Kindle uses type params: `Linear<s![784,128], B>`").
- **Cite the ground-truth file** so the agent can verify, not hallucinate.
- **End with the verification command** to run (`cargo kindle check`).
- Keep each under ~150 lines — a skill is a *cheat sheet*, not the book (link to
  the book, doc `07`, for depth).

## The skill set (build these, in this order)

### Task 08.1 — the core three (highest error-rate areas)
1. **`kindle-new-model`** — `#[module]` structs, static shape params, forward
   signature (`Tensor<s![dyn, 784], B> -> Tensor<s![dyn, 10], B>`). Ground truth:
   `README.md` MLP example, `kindle-macros/src/module.rs`, `nn/module.rs`.
   Top mistakes: constructor-style layer init; forgetting `<B: Backend>`;
   wrong `dyn` batch placement.
2. **`kindle-fix-shape-errors`** — how to read a shape compile error and fix it.
   **This is the killer skill:** instruct the agent to run `cargo kindle check`
   (humanized) rather than raw `cargo check`, read the decimal shapes, and adjust
   the `s![…]` params. Ground truth: doc `01`/`02`, `kindle-diagnostics`.
   This turns Kindle's *strength* (compile-time errors) into an agent
   *superpower* — the agent gets a precise, machine-readable "you need 128 not
   784" instead of guessing from a runtime stack trace.
3. **`kindle-training-loop`** — the canonical loop: `DataLoader` iteration,
   `AdamW::new(model.parameters(), lr)`, forward, loss, `backward`, telemetry.
   Ground truth: `examples/mnist_training.rs`, `optim/`, doc `05`. Top mistakes:
   looking for `zero_grad()` (there is none — say so explicitly); manual gradient
   plumbing.

### Task 08.2 — the interop/deploy three
4. **`kindle-load-pretrained`** — `kindle::hub` + `load_safetensors` +
   `from_pretrained`. Ground truth: `nn/save`, `kindle::hub`, `IDEAS.md`.
5. **`kindle-export`** — GGUF/MLX/safetensors export, `cargo kindle inspect`.
   Ground truth: `io/`, `export_test.rs`, doc `06`. Note which quant schemes are
   actually supported (currently F32 + Q8_0) so the agent does not request an
   unimplemented one.
6. **`kindle-backends`** — feature flags, `KindleBackend<T, D>`, `TransferTo`
   between CPU/CUDA/WGPU. Ground truth: `kindle-backends`, `examples/backends`.

### Task 08.3 — the meta skills
7. **`kindle-verify`** — the exact verification loop (README §2). An agent that
   loads this runs the *repo's* fmt/clippy/test commands, not guessed ones.
8. **`kindle-shapes-and-slicing`** — `s!`, `idx!`, named dims, reshape rules.
   Ground truth: `shapes/*`, `examples/idx_demo`, doc `03`.

Each skill directory may include a `reference/` subdir (e.g. the Rosetta table
from doc `07` Appendix A as `reference/pytorch-rosetta.md`) and, where useful, a
`scripts/` helper — but keep the `SKILL.md` itself lean.

### Task 08.4 — the installer: `cargo kindle skills install`
Add a `skills` subcommand to `cargo-kindle.rs`:
- `cargo kindle skills install [--agent claude|cursor|all] [--dest <dir>]` —
  detect the agent tool (presence of `.claude/`, `.cursor/`, etc.) and copy
  `skills/*` into its skills/rules directory; default `--agent all`.
- `cargo kindle skills list` — print available skills + descriptions.
- Embed the skill sources with `include_dir!`/`include_str!` so the CLI carries
  them (works even when installed via `cargo install`, away from the repo).
- **Idempotent & non-destructive:** never overwrite a user-modified skill
  without `--force`; report what it wrote.

### Task 08.5 — root `AGENTS.md`
A committed, tool-neutral `AGENTS.md` at repo root that any agent reads first:
- one-paragraph "what Kindle is and its one differentiator";
- the **non-negotiable idioms** (static shape params, no `zero_grad`, run
  `cargo kindle check` for readable errors, the verification loop);
- a pointer to `cargo kindle skills install` and to the book;
- the top-5 mistakes-to-avoid, condensed from the skills.
This single file is the cheapest, highest-reach agent artifact — it works in
tools that have no skill system at all.

## Verification
- `cargo kindle skills list` shows all 8; `cargo kindle skills install --dest
  /tmp/x` writes them and is idempotent on a second run.
- **Dogfood test (the real acceptance):** in a *scratch* project, give a fresh
  agent only `AGENTS.md` + the installed skills and the task "train an MLP on
  MNIST and export it to GGUF." It should produce compiling code **without**
  searching the web or guessing the API. Record the transcript; every place the
  agent still had to guess is a gap to fix in the relevant skill. Iterate until
  the guesswork is gone — that iteration *is* the deliverable.
- Skills must stay in sync: add a CI check that every file path a skill cites
  still exists (grep the `Ground truth:` lines).

## Risks / DO-NOT
- **DO-NOT** author skills inside `.claude/` — it is gitignored; author under
  `skills/` and install into `.claude/`.
- **DO-NOT** write generic filler ("Kindle is a great framework…"). A skill
  earns its place only by removing a *specific* guess. If a section does not stop
  a concrete mistake, cut it.
- **DO-NOT** let skills document unshipped APIs — same rule as the book. Skills
  ship *with* the feature.
- **DO-NOT** cite line numbers in skills (they rot fastest) — cite file paths +
  symbol names, and let the agent grep.

## Demo script
Split screen: two identical agents, same prompt ("build and train an MNIST
classifier, export to GGUF"). One with plain PyTorch, one with Kindle + installed
skills. The Kindle agent writes compiling, shape-checked code first try, the
compiler catches its one mistake instantly, and it exports a running GGUF — no
web searches. Caption: *"The agent didn't guess once. The library told it, and
the compiler checked it."*
